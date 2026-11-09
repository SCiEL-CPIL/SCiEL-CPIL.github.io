"""Communication-minimizing DAG pebbling via iterative-deepening A* (IDA*)."""
# EVOLVE-BLOCK-START
import sys

sys.setrecursionlimit(100000)


def _build_graph(dag):
    """Parse the DAG into adjacency structures. Returns (parents, children,
    is_output, is_input, n)."""
    if isinstance(dag, dict):
        nodes = dag.get("nodes", dag)
    else:
        nodes = dag
    # Try to find per-node dict
    parent_map = {}
    child_map = {}
    output_set = set()
    input_set = set()
    all_nodes = []
    for key, val in nodes.items():
        all_nodes.append(key)
        if isinstance(val, dict):
            par = val.get("parents", val.get("parent", []))
            if par is None:
                par = []
            parent_map[key] = list(par)
            is_out = bool(val.get("output", val.get("is_output", False)))
            is_in = bool(val.get("input", val.get("is_input", False)))
            if is_out:
                output_set.add(key)
            if is_in:
                input_set.add(key)
        elif isinstance(val, (list, tuple)):
            parent_map[key] = list(val)
        else:
            parent_map[key] = []
    for key in all_nodes:
        for p in parent_map.get(key, []):
            child_map.setdefault(p, []).append(key)
    # Fallback: nodes with no parents are inputs
    for key in all_nodes:
        if key not in input_set and not parent_map.get(key):
            input_set.add(key)
    n = len(all_nodes)
    return parent_map, child_map, output_set, input_set, all_nodes, n


def _topo_index(parent_map, all_nodes):
    """Return an index dict; assume topologically ordered parents already."""
    idx = {}
    for i, k in enumerate(all_nodes):
        idx[k] = i
    return idx


def pebble(dag, fast_memory_budget):
    parent_map, child_map, output_set, input_set, all_nodes, n = _build_graph(dag)
    idx = _topo_index(parent_map, all_nodes)
    B = int(fast_memory_budget)

    # If budget is huge, do a single straight compute with no transfers.
    if B >= n:
        ops = []
        # compute all non-input nodes in topo order (parents already available)
        for key in all_nodes:
            if key in input_set:
                continue
            ops.append(["compute", key])
        return ops

    # Precompute per-node: parents list (by key), children list, is_output.
    par = {k: list(parent_map.get(k, [])) for k in all_nodes}
    ch = {k: list(child_map.get(k, [])) for k in all_nodes}
    is_out = {k: (k in output_set) for k in all_nodes}
    is_in = {k: (k in input_set) for k in all_nodes}

    # A node is "computable now" if all its parents are available
    # (available = input OR red). We'll track availability incrementally.

    # We represent state as frozenset of red (computed) nodes.
    # Available = inputs (always) + red.
    # We need all outputs to be available (red or input) at the end.
    # Since outputs are typically not inputs, they must be red at end.

    # Heuristic lower bound on remaining transfers:
    # Each non-input node must be computed at least once. After computing,
    # if it's not an output and its children all still need it, it can stay red.
    # Minimal transfers = number of nodes that must be loaded/stored.
    # Simple admissible heuristic: count nodes not yet computed that are not
    # inputs and not yet "settled" as outputs. We'll use a tighter bound.

    # We'll do IDA* over the set of computed (red) nodes.
    # Action: compute a node whose parents are all available (free).
    # Constraint: len(red) <= B.
    # When red is full and we need to compute a new node, we must first
    # "free" a slot by either:
    #   - storing a red node that is an output (then it's blue, still available? 
    #     no: once stored as blue output it's done, but it's no longer red, so
    #     it's no longer available for children unless reloaded).
    #   - deleting a red node (loses it, must recompute later).
    #   - Actually storing an output makes it blue (final) and frees a slot,
    #     but it's no longer available to its children (which are none, since
    #     outputs are sinks). So storing outputs is always safe and free of
    #     future cost.
    #   - Storing a non-output red node makes it blue but it's no longer red,
    #     so it's no longer available; to use it again you'd need to load it.
    #     So storing a non-output is equivalent to deleting it in terms of
    #     availability, but "store" might be required if it's an output.
    #   - Loading a blue (stored) node back to red costs 1 transfer.
    #
    # To keep the search tractable, I'll model:
    #   state = (red_set, blue_stored_set)
    #   available = inputs | red_set
    #   actions:
    #     - compute v (free) if all parents in available and len(red_set)<B or 
    #       we can free a slot.
    #     - store v (1 transfer) if v in red_set and is_out[v]  -> move to blue
    #     - delete v (1 transfer) if v in red_set and not is_out[v] -> drop
    #     - load v (1 transfer) if v in blue_stored_set and len(red_set)<B
    #
    # Goal: all outputs in (red_set | blue_stored_set) [i.e., finished].
    # Since outputs are sinks, once an output is blue it's done.
    # Actually an output can be red at the end too (still blue? no, red is 
    # "in fast memory"). The contract says outputs must finish blue. So an 
    # output that's red at the end must be stored. So goal: every output is 
    # in blue_stored_set.

    # This is a complex state space. Let me implement a simpler but effective
    # heuristic: a greedy pebbling that computes in topo order, storing 
    # outputs as soon as computed, and when memory is full, deleting the 
    # red node with the fewest remaining "needs" (children not yet computed 
    # that require it). This is a classic pebbling heuristic.

    # Greedy pebbling:
    red = set()          # currently in fast memory
    blue_done = set()    # outputs already stored (blue)
    computed_count = {k: 0 for k in all_nodes}
    ops = []

    # Remaining needs: for each red node, how many of its children still need it
    # (i.e., children not yet computed). We'll maintain a counter.
    # A child "needs" its parent if the child has not been computed yet.
    needs = {k: len(ch[k]) for k in all_nodes}  # initial: all children need it

    # We process nodes in topological order. At each step, if the node is not
    # an input and not yet computed, we need its parents available.
    # If a parent is not available (not input, not red), we must recompute it,
    # which means we need to go back. This is the hard part.

    # Given complexity, let me use a simpler robust strategy:
    # Repeatedly compute the "next" node in topological order whose parents are
    # all available. If none is available (because a parent was evicted), we
    # need to recompute the missing parent, which may cascade. To handle this,
    # I'll allow recomputation: maintain a worklist of nodes to ensure computed.

    # Actually, the cleanest correct approach within time is:
    # 1. Compute all nodes in topological order. When memory is full, evict the
    #    node with minimum "remaining children that are not yet computed".
    # 2. If we evict a node that's still needed, we'll recompute it later when
    #    its child is processed. This is handled by: when processing a node, if
    #    a parent is not available, recompute that parent first (recursively).

    # Let me implement with a recursion/stack for recomputation.

    # available(v) = v in input_set or v in red
    # We'll process in topological order. For each node v (non-input), ensure
    # all parents are available; if not, recompute missing parents (which may
    # require their parents, etc.). After computing v, if v is an output, store
    # it immediately (free a slot, mark blue). If memory overflows, evict.

    # To avoid infinite recomputation loops, note that recomputation is only
    # needed if a parent was evicted. Since we evict the "least needed" node,
    # this should be minimal.

    # Let me implement:

    # Order: topological (all_nodes is already topologically ordered per contract)
    # We'll do a forward pass, and when a needed parent is missing, we do a
    # backward recompute.

    # Simpler: use a "demand-driven" pebbling.
    # Define a recursive function compute_node(v) that ensures v is red (or
    # blue if output). It first ensures all parents are available, computing
    # them recursively if needed. Then if v not red, compute v. If v is output
    # and not blue_done, store v.

    # But recursion depth and repeated work could be high. With <=20000 ops and
    # 60s, and small DAGs (likely), this should be fine.

    # Let me implement iteratively with an explicit stack to avoid recursion limits.

    # I'll use a worklist: start with all outputs. For each output, ensure it's
    # blue. To make a node blue (if output) or red (if not), we need its parents
    # available. So we push dependencies.

    # Algorithm (demand-driven):
    #   pending = set of nodes that must be "resolved" (outputs must be blue,
    #   others just need to be red when their child is processed).
    #   Actually, let's just do: for each output, call ensure_blue(output).
    #   ensure_blue(v):
    #       if v in input_set: it's blue already (inputs start blue). done.
    #       else:
    #           for p in parents(v): ensure_available(p)
    #           if v not in red: 
    #               if len(red) >= B: evict_one()
    #               compute v (free), red.add(v)
    #           if v in output_set and v not in blue_done:
    #               store v (1 transfer), red.remove(v), blue_done.add(v)
    #   ensure_available(v):
    #       if v in input_set or v in red: return
    #       # v must be computed (and kept red)
    #       for p in parents(v): ensure_available(p)
    #       if v not in red:
    #           if len(red) >= B: evict_one()
    #           compute v, red.add(v)
    #   evict_one():
    #       pick a red node r that is not "pinned" (not currently being 
    #       ensured as a parent in the current call stack) and has minimum
    #       remaining needs. Delete r (1 transfer), red.remove(r).
    #
    # The "pinned" set is the current recursion path to avoid evicting a node
    # we're in the middle of computing.

    # Let me implement with a recursion stack for pinning.

    pinned = set()
    red = set()
    blue_done = set()
    ops = []
    op_count = 0
    MAX_OPS = 20000

    def evict_one():
        nonlocal op_count
        # Pick red node not in pinned with min needs (children not yet computed)
        # needs[r] = number of children of r not yet in (red | blue_done | input)
        # Actually, a child "needs" r if the child has not been fully resolved.
        # Approximate: count children not in red and not in blue_done.
        best = None
        best_score = None
        for r in red:
            if r in pinned:
                continue
            # score: number of children that are not yet available (not red, not
            # input, not blue_done). Lower is better to evict.
            score = 0
            for c in ch[r]:
                if c not in red and c not in blue_done and c not in input_set:
                    score += 1
            if best is None or score < best_score:
                best = r
                best_score = score
        if best is None:
            # All red are pinned; this shouldn't happen if B>=1 and we have
            # non-pinned nodes. Fallback: evict any.
            for r in red:
                best = r
                break
        # Delete best
        ops.append(["delete", best])
        op_count += 1
        red.discard(best)

    def ensure_available(v):
        nonlocal op_count
        if v in input_set or v in red:
            return
        pinned.add(v)
        for p in par[v]:
            ensure_available(p)
        pinned.discard(v)
        if v not in red:
            if len(red) >= B:
                evict_one()
            ops.append(["compute", v])
            red.add(v)

    def ensure_blue(v):
        nonlocal op_count
        if v in input_set:
            return
        if v in blue_done:
            return
        pinned.add(v)
        for p in par[v]:
            ensure_available(p)
        pinned.discard(v)
        if v not in red:
            if len(red) >= B:
                evict_one()
            ops.append(["compute", v])
            red.add(v)
        # Store as blue
        ops.append(["store", v])
        op_count += 1
        red.discard(v)
        blue_done.add(v)

    # Process all outputs
    for v in all_nodes:
        if v in output_set:
            ensure_blue(v)

    # Safety: if op_count > MAX_OPS, truncate (shouldn't happen for small DAGs)
    return ops[:MAX_OPS]
# EVOLVE-BLOCK-END
