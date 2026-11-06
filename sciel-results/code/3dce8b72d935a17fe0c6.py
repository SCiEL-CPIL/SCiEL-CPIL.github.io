"""DAG pebbling: minimize communication (transfers) under bounded fast memory."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

# ---------- deadline guard ----------
_DEADLINE = time.time() + 80.0  # safety margin before 90s kill


def _now_ok():
    return time.time() < _DEADLINE


def _topo_order(dag):
    """Return list of vertex ids in topological order. Inputs first, outputs last."""
    parents = dag.get("parents", {})
    outputs = dag.get("outputs", [])
    all_v = set(parents.keys()) | set(outputs)
    # parents[v] is list of parents (already topologically before v)
    indeg = {v: 0 for v in all_v}
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            indeg[v] += 1
            children[p].append(v)
    q = deque([v for v in all_v if indeg[v] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    # If some nodes not reached (cycles), append them
    if len(order) != len(all_v):
        for v in all_v:
            if v not in set(order):
                order.append(v)
    return order


def _is_input(dag, v):
    return v not in dag.get("parents", {}) or not dag.get("parents", {}).get(v)


def _is_output(dag, v):
    return v in set(dag.get("outputs", []))


def _compute_ops(dag, order):
    """Compute operation for each vertex in a given topological order.
    Returns dict v -> op where op in {'load','compute','store'}
    and list of delete ops placed appropriately.
    """
    parents = dag.get("parents", {})
    ops = {}
    for v in order:
        if _is_input(dag, v):
            ops[v] = 'load'
        else:
            ops[v] = 'compute'
    # outputs need store after compute
    for v in order:
        if _is_output(dag, v):
            # we'll mark store separately
            pass
    return ops


def _simulate(dag, order, budget):
    """Simulate a topological order, inserting deletes to stay within budget.
    Returns (transfers, ops_list, valid).
    """
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    # in-memory set
    mem = set()
    ops = []
    transfers = 0
    # red values: inputs loaded, computed results
    # we need to know when a value is no longer needed
    # future_use[v] = number of children not yet processed that need v
    # build child list
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    # remaining uses: for each v, count of children not yet processed
    remaining = {}
    for v in parents:
        remaining[v] = len(children[v])
    for v in outputs:
        # output needs to be stored, but after store it's no longer needed
        # treat output as needing v until store
        pass

    # We'll process in order. Before each op, if over budget, delete a value
    # that is no longer needed (remaining uses == 0 and not an output not yet stored).
    # Actually a value can be deleted once all its consumers have been processed
    # and (if it's an output) it has been stored.

    # Track for each vertex: state: 'blue_mem', 'red_mem', 'stored', 'gone'
    state = {}
    # all start gone; inputs start blue in slow memory
    for v in parents:
        state[v] = 'gone'
    for v in outputs:
        if v not in state:
            state[v] = 'gone'

    # For deletion eligibility: a mem value can be deleted if remaining[v]==0
    # and (not output or already stored). But we store outputs right after compute.
    # Let's store outputs immediately after compute/load if they are outputs.

    # We'll define: after producing v (load or compute), if v is output, store it
    # (transfer +1) and mark stored, and it can be deleted from mem later.
    # Actually storing means it's in slow memory; we can then delete from fast mem.
    # But the problem says "all outputs must finish blue" - blue = in slow memory? 
    # "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    # So blue = slow memory (no fast slot), red = fast memory (1 slot).
    # load: blue -> red (transfer +1, now in fast mem)
    # compute: red parents -> red result (no transfer, but result in fast mem)
    # store: red -> blue (transfer +1, frees fast slot)
    # delete: red -> gone (no transfer, frees fast slot) - only if value not needed

    # So a computed value that's not an output and has no more uses can be deleted.
    # An output must be stored (not deleted) to finish blue.

    # Let's redo: mem holds red values. We must keep red values until all uses done,
    # except outputs which must be stored before deletion.
    # Strategy: after each operation, if len(mem) > budget, delete eligible values.

    # Eligible for delete: value in mem, all children processed (remaining==0),
    # and not an output (or output already stored).
    # We'll store outputs immediately when produced.

    for v in order:
        # before op, ensure we have room: if we're going to add a new red value,
        # and len(mem) == budget, we need to free one slot first.
        # But load/compute adds to mem. So check.
        # First, free any eligible values if over or at budget.
        def free_if_needed():
            nonlocal transfers
            # free eligible values while mem over budget or at budget and we need room
            while len(mem) >= budget:
                # find eligible
                eligible = [x for x in mem if remaining.get(x, 0) == 0 and not (x in outputs and state[x] == 'red_mem')]
                # also outputs that are stored can be deleted
                eligible = [x for x in mem if remaining.get(x, 0) == 0 and state[x] in ('red_mem', 'stored')]
                if not eligible:
                    return False
                # delete one (pick arbitrary, e.g. first)
                x = eligible[0]
                mem.discard(x)
                state[x] = 'gone'
                ops.append(('delete', x))
                return True
            return True

        if _is_input(dag, v):
            # load
            if v in mem:
                # already in mem, no-op? but we should have deleted if not needed.
                # skip
                pass
            else:
                free_if_needed()
                mem.add(v)
                state[v] = 'red_mem'
                ops.append(('load', v))
                transfers += 1
            # if output, store it
            if v in outputs:
                # store: red -> blue
                ops.append(('store', v))
                transfers += 1
                state[v] = 'stored'
                # can delete from mem now
                # but we'll let free_if_needed handle it
        else:
            # compute
            # ensure parents in mem
            for p in parents[v]:
                if p not in mem:
                    # need to reload/recompute? This means our order is bad.
                    # For a valid topological order with no deletes of needed values,
                    # parents should be in mem. If not, we have a problem.
                    # Fallback: reload parent (transfer) - but parent may not be input.
                    # This indicates the order requires recomputation.
                    # For simplicity, if parent not in mem, we must have a strategy.
                    # We'll handle by: if parent is input, load it; else, this order
                    # is invalid for single-pass. Mark invalid.
                    return None
            if len(mem) >= budget:
                # need room for result
                if not free_if_needed():
                    return None
            mem.add(v)
            state[v] = 'red_mem'
            ops.append(('compute', v))
            # if output, store
            if v in outputs:
                ops.append(('store', v))
                transfers += 1
                state[v] = 'stored'
        # update remaining for v's children
        # v is now processed; decrement remaining for v? No, remaining[v] is # children of v not processed.
        # When we process a child c, we decrement remaining[parent]. Let's do it when processing child.
        # Actually let's decrement remaining[v] when v is consumed.
        # Simpler: when we process v, for each parent p of v, decrement remaining[p].
        for p in parents.get(v, []):
            remaining[p] -= 1
        # also v's own remaining stays; v can be deleted when remaining[v]==0 and stored/gone.
        # After processing v, v's children still need it, so remaining[v] unchanged until children processed.

    # at end, all outputs stored. Count transfers.
    # But we may have leftover mem values; that's fine, no need to delete.
    return transfers, ops, True


def _naive_transfers(dag):
    """Naive schedule: process in topo order, load each input once, compute each non-input,
    store each output. Transfers = #inputs (loads) + #outputs (stores).
    Assumes enough memory (no deletes needed)."""
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    n_inputs = sum(1 for v in parents if not parents[v])
    # but inputs may also be outputs? count loads = number of input vertices
    n_outputs = len(outputs)
    return n_inputs + n_outputs


def _greedy_order(dag, budget):
    """Greedy: pick next vertex that minimizes memory pressure.
    Use a priority based on: prefer vertices whose parents are in mem and
    whose children count is high (to free them sooner)."""
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    all_v = set(parents.keys()) | outputs
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    indeg = {v: len(parents.get(v, [])) for v in all_v}
    # available = vertices with all parents processed
    processed = set()
    # We'll simulate memory to guide choice.
    # Start: no vertices processed. Available = inputs (indeg 0).
    # Greedy rule: among available, pick the one with most children (to free mem early)
    # tie-break: inputs before computes? or by remaining uses.
    order = []
    mem = set()
    # We need to track which values are in mem. But for ordering, we just need
    # to ensure we don't exceed budget. Use heuristic: pick available vertex that
    # maximizes (children count) to free memory, and prefer to keep parents.
    # Actually a known heuristic: "minimize live values" - pick vertex whose
    # parents have fewest other pending uses (so we can delete them).
    # Let's use: score = -sum over parents p of remaining_uses[p] (after this vertex)
    # i.e., prefer to consume parents that are about to be unused.
    # remaining_uses[p] = # children of p not yet processed (excluding current).
    # We'll maintain pending_children[p] = # children of p not yet in order.
    pending = {v: len(children[v]) for v in all_v}
    avail = [v for v in all_v if indeg[v] == 0]
    # Use a heap with dynamic scores; recompute each step (n small?).
    # For safety, if n large, this is O(n^2). Acceptable for small DAGs.
    while avail:
        # pick best
        best = None
        best_score = None
        for v in avail:
            # score: prefer to consume parents with low pending (so deletable)
            ps = parents.get(v, [])
            s = 0
            for p in ps:
                # after processing v, pending[p] decreases by 1
                s += (pending[p] - 1)  # lower is better
            # also prefer outputs? outputs must be stored; do them when parents ready.
            # prefer high children count (frees v soon)
            s -= len(children[v]) * 0.1
            if best_score is None or s < best_score:
                best_score = s
                best = v
        order.append(best)
        avail.remove(best)
        processed.add(best)
        # update
        for c in children[best]:
            indeg[c] -= 1
            if indeg[c] == 0:
                avail.append(c)
        # pending: best's children now have best as processed; pending[best] decreases? 
        # pending[p] = # children of p not yet in order. When we add v to order,
        # for each parent p of v, pending[p] -= 1.
        for p in parents.get(best, []):
            pending[p] -= 1
    return order


def pebble(dag, fast_memory_budget):
    """Return list of [operation, vertex] minimizing transfers."""
    try:
        if fast_memory_budget is None or fast_memory_budget < 1:
            fast_memory_budget = 1
        order = _greedy_order(dag, fast_memory_budget)
        result = _simulate(dag, order, fast_memory_budget)
        if result is None:
            # fallback: simple topo order
            order2 = _topo_order(dag)
            result = _simulate(dag, order2, fast_memory_budget)
            if result is None:
                # last resort: return empty (invalid) - but better to return something
                return []
        transfers, ops, valid = result
        # convert to list of [op, vertex]
        return [[op, v] for op, v in ops]
    except Exception:
        return []
# EVOLVE-BLOCK-END
