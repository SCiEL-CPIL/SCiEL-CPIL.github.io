"""Rewritten program for communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Produce a pebbling schedule that minimizes the mean number of transfers
    relative to the naive schedule across layered DAGs.

    dag: object with attributes:
         - vertices: list of vertex ids (topologically ordered)
         - parents: dict vertex -> list of parent vertex ids (already topo-ordered)
         - outputs: list of output vertex ids
         - inputs: list of input vertex ids (start blue)
         Possibly also: - children: dict vertex -> list of children
    fast_memory_budget: int, number of red slots available.

    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}.
    """
    # --- extract structure defensively ---
    vertices = getattr(dag, 'vertices', None)
    if vertices is None:
        vertices = list(getattr(dag, 'nodes', []))
    parents = getattr(dag, 'parents', None)
    if parents is None:
        parents = getattr(dag, 'preds', {})
    outputs = set(getattr(dag, 'outputs', []))
    inputs = set(getattr(dag, 'inputs', []))

    # Build children map
    children = {}
    for v in vertices:
        children.setdefault(v, [])
    for v, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(v)

    # Identify leaves (no children) and roots (no parents)
    leaves = [v for v in vertices if len(children.get(v, [])) == 0]
    roots = [v for v in vertices if len(parents.get(v, [])) == 0]

    # Determine which vertices are inputs
    input_set = inputs
    if not input_set:
        input_set = set(roots)

    # A vertex is "available" to compute when all its parents are blue.
    # We use a greedy strategy: 
    #   1. Load all inputs.
    #   2. Compute vertices in topological order, but be smart about memory.
    #   3. The key optimization: instead of storing every computed value,
    #      we recompute when necessary if memory is tight.
    #
    # A simpler and often effective strategy for layered DAGs:
    #   - Process layer by layer.
    #   - For each layer, compute all vertices in that layer.
    #   - After processing a layer, decide which values to keep in fast memory
    #     based on future need (how many times each value is used in remaining layers).
    #
    # Let's implement a reference-counting based memory manager.

    # Count how many times each vertex is used as a parent in the remaining DAG
    # (i.e., number of children that haven't been processed yet).
    # We'll track remaining_children dynamically.

    remaining_children = {}
    for v in vertices:
        remaining_children[v] = len(children.get(v, []))

    # Also track in-degree (uncomputed parents)
    pending_parents = {}
    for v in vertices:
        pending_parents[v] = len(parents.get(v, []))

    # State: blue_set (vertices currently blue), red_set (vertices in fast memory as red)
    # Actually, in pebbling: blue = on tape (infinite), red = in fast memory (bounded).
    # Initially all inputs are blue. We need all outputs blue at the end.
    # Operations:
    #   load: blue -> red (costs 1 transfer, uses 1 fast slot)
    #   store: red -> blue (costs 1 transfer, frees 1 fast slot)
    #   compute: takes red parents, produces red child (costs 0 transfers, but uses 1 slot)
    #   delete: red -> nothing (frees 1 slot, no transfer cost)

    # Wait, re-reading: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot, including newly computed results."
    # "Score mean transfers / naive schedule"
    # Transfers = load + store operations. Compute and delete don't count as transfers?
    # The contract says "Score mean transfers", so we minimize load+store count.

    # Let me re-read: "operations load/store/compute/delete"
    # I'll assume:
    #   load: from blue to red (transfer)
    #   store: from red to blue (transfer)
    #   compute: from red parents to red child (no transfer)
    #   delete: remove red (no transfer)

    # Strategy:
    # 1. Initially, all inputs are blue. We need them red to compute.
    # 2. We'll maintain a set of red vertices (in fast memory) and a set of blue vertices.
    # 3. We process vertices in topological order.
    # 4. For each vertex to compute, we need all its parents to be red.
    # 5. If a parent is blue, we load it (transfer).
    # 6. After computing, we decide whether to keep the result red or store it blue or delete it.
    # 7. Decision heuristic: keep red if remaining_children[v] > 0 and we have room,
    #    otherwise store to blue if it's an output or will be needed again, else delete.

    # Actually, a better heuristic for layered DAGs with limited memory:
    #   - Keep a vertex red if it has multiple children in the remaining graph
    #     (i.e., high reuse), provided we have memory slots.
    #   - If memory is full, evict the vertex with the lowest future reuse.

    # Let's implement a simpler greedy approach that works well for typical layered DAGs:

    red = set()       # vertices currently in fast memory
    blue = set(input_set)  # vertices currently on blue tape
    # We'll also track which vertices have been computed
    computed = set()

    ops = []

    # Topological order is already given by vertices list
    # But we need to process in an order that allows computation.

    # For each vertex in topo order:
    for v in vertices:
        # Check if v is an input (already blue, no need to compute)
        if v in input_set:
            # Input: it's blue. We need it red to compute children.
            # But we don't compute it; it's just available.
            # We'll load it when needed.
            continue

        # v needs to be computed. All its parents must be red.
        ps = parents.get(v, [])
        for p in ps:
            if p not in red:
                # p is blue (must be, since it was computed earlier)
                # Load p into red
                if len(red) >= fast_memory_budget:
                    # Need to free a slot. Evict a red vertex with lowest future use.
                    # Find red vertex with minimum remaining_children (excluding those already processed)
                    # Actually, we should evict the one with least future reuse.
                    evict_candidates = [r for r in red if r != v]
                    if evict_candidates:
                        # Pick the one with min remaining_children
                        best_evict = min(evict_candidates, key=lambda r: remaining_children.get(r, 0))
                        # Store it back to blue (transfer) or delete?
                        # If it's an output or will be needed, store. Else delete.
                        # For safety, store if it might be needed again, else delete.
                        if remaining_children.get(best_evict, 0) > 0 or best_evict in outputs:
                            ops.append(['store', best_evict])
                            blue.add(best_evict)
                        else:
                            ops.append(['delete', best_evict])
                        red.discard(best_evict)
                    else:
                        # Can't evict, but this shouldn't happen if budget >= 1
                        # Force store of any red
                        for r in list(red):
                            ops.append(['store', r])
                            blue.add(r)
                            red.discard(r)
                            break

                ops.append(['load', p])
                red.add(p)
                blue.discard(p)

        # Now all parents are red. Compute v.
        ops.append(['compute', v])
        red.add(v)
        computed.add(v)

        # Update remaining_children for parents (they've been "used" for v)
        for p in ps:
            if p in remaining_children:
                remaining_children[p] -= 1

        # Decide what to do with v: keep red, store blue, or delete
        # Heuristic:
        #   - If v is an output, we must eventually make it blue. Keep red for now, store at end.
        #   - If remaining_children[v] > 0, it will be needed again. Keep red if we have room.
        #   - If remaining_children[v] == 0 and v not in outputs, we can delete or store.
        #     Delete is free, store costs a transfer. Delete if not output.
        #   - But if memory is tight, we might need to free v early.

        # For now, keep v red. We'll handle eviction when we need more space.
        # But if remaining_children[v] == 0 and v not in outputs, we could delete immediately.
        if remaining_children.get(v, 0) == 0 and v not in outputs:
            # No future use, not an output. Delete to free a slot.
            ops.append(['delete', v])
            red.discard(v)
        # Else keep red for now.

    # After processing all vertices, all outputs must be blue.
    for o in outputs:
        if o in red:
            ops.append(['store', o])
            blue.add(o)
            red.discard(o)
        elif o not in blue:
            # Shouldn't happen, but ensure it's blue
            # If o was deleted, we'd need to reload... but we shouldn't delete outputs.
            pass

    # Ensure inputs that were loaded and stored are handled correctly.
    # The ops list should be valid.

    # Truncate if too many operations (safety)
    if len(ops) > 20000:
        # This shouldn't happen for reasonable DAGs, but truncate as safety
        # Actually, we need to ensure outputs are blue. Let's not truncate blindly.
        # Instead, we'll rely on the algorithm being efficient.
        pass

    return ops

# EVOLVE-BLOCK-END
