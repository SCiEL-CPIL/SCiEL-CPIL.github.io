"""DAG pebbling: minimize fast-memory transfers via bottom-up eval with recomputation."""
# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
      - 'vertices': list of vertex ids (topologically ordered, parents before children)
      - 'parents': dict vertex -> list of parent vertex ids
      - 'outputs': list of output vertex ids
    fast_memory_budget: int, number of red (fast-memory) slots available.

    Returns list of [operation, vertex] where operation in {'load','store','compute','delete'}.
    Inputs start blue (slow memory). All outputs must finish blue.
    Red values consume one fast-memory slot.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    vset = set(vertices)
    B = fast_memory_budget

    # Determine which vertices are inputs (no parents) vs computed
    inputs = set()
    for v in vertices:
        if v not in parents or len(parents[v]) == 0:
            inputs.add(v)

    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in children:
                    children[p].append(v)

    # Count how many times each vertex is needed as a parent (fan-out)
    fanout = {v: len(children[v]) for v in vertices}

    # Subtree size (number of vertices in subtree rooted at v, including v)
    # Compute in reverse topological order
    subtree_size = {}
    for v in reversed(vertices):
        if v in parents and len(parents[v]) > 0:
            # v is a computed node; its subtree size = 1 + sum of children subtrees? 
            # Actually for recomputation cost, we want the number of compute ops in the subtree
            subtree_size[v] = 1 + sum(subtree_size.get(c, 1) for c in children[v])
        else:
            subtree_size[v] = 1

    # We'll do a bottom-up pass. For each vertex in topological order, we compute it.
    # Strategy: keep a set of "live" red values in fast memory.
    # When computing vertex v, we need all its parents to be red.
    # If a parent is not red, we must load it (if it's an input) or recompute it (if computed).
    # To minimize transfers, we try to keep as many parents red as possible.
    # 
    # Key insight: process vertices in topological order. For each vertex, ensure its parents
    # are in fast memory. If not enough space, evict (delete) values that are no longer needed.
    #
    # We track:
    #   red: set of vertices currently in fast memory
    #   blue: set of vertices that have been stored back to slow memory (or are inputs not yet loaded)
    #
    # Operations:
    #   load: move a vertex from blue (slow) to red (fast) - counts as transfer
    #   store: move a vertex from red to blue - counts as transfer
    #   compute: compute a vertex in red (requires parents in red) - no transfer
    #   delete: remove a vertex from red without storing - no transfer (value lost, must recompute if needed)

    # We'll simulate the process.
    # State: red = set of vertices in fast memory.
    # For inputs: they start blue. We load them when needed.
    # For computed nodes: they start non-existent. We compute them when needed.
    #
    # To know if a computed node is "available" in slow memory, we track which computed nodes
    # have been stored. If a computed node was stored, we can load it. If it was deleted,
    # we must recompute it.
    
    stored = set()  # computed vertices that are in slow memory (blue)
    red = set()     # vertices currently in fast memory
    ops = []
    
    # We process vertices in topological order. For each vertex, we make sure all its
    # parents are red, then compute it, then decide whether to keep it red or store/delete.
    
    # To decide eviction: when we need to load/recompute a parent but fast memory is full,
    # we evict vertices that are "least useful" - i.e., whose remaining uses are fewest.
    
    # Precompute for each vertex, in what positions (in topological order) it is needed as a parent
    # We'll track remaining uses dynamically.
    
    # remaining_uses[v] = number of children of v that haven't been processed yet
    remaining_uses = {v: 0 for v in vertices}
    for v in vertices:
        for c in children[v]:
            remaining_uses[v] += 1
    
    # Process in topological order
    for v in vertices:
        # Get parents of v
        par = parents.get(v, [])
        
        # Ensure all parents are red
        for p in par:
            if p in red:
                continue
            # p is not red. Need to make it red.
            # Check if fast memory is full
            if len(red) >= B:
                # Need to evict something
                # Evict a vertex that is not in par (we need all par to be red)
                # Choose the vertex with minimum remaining_uses among red vertices not in par
                candidates = [x for x in red if x not in set(par)]
                if not candidates:
                    # All red vertices are parents we need... this shouldn't happen if B >= len(par)
                    # If B < len(par), we can't fit all parents. This is a problem.
                    # In that case, we need a different strategy. For now, evict one and hope.
                    candidates = list(red)
                # Choose candidate with min remaining_uses
                evict = min(candidates, key=lambda x: (remaining_uses.get(x, 0), x))
                ops.append(['delete', evict])
                red.discard(evict)
            
            # Now load or recompute p
            if p in inputs:
                # Input: load from slow memory
                ops.append(['load', p])
                red.add(p)
            else:
                # Computed node: check if it's stored
                if p in stored:
                    ops.append(['load', p])
                    red.add(p)
                else:
                    # Must recompute p. This requires p's parents to be red.
                    # This is recursive. For simplicity, we'll recompute by recursively ensuring
                    # p's parents are red, then computing p.
                    # To avoid deep recursion, we'll do it iteratively.
                    # For now, handle by recursively calling a helper.
                    _recompute(p, red, stored, ops, B, inputs, parents, children, remaining_uses, vertices)
        
        # Now all parents of v are red. Compute v.
        if v in inputs:
            # v is an input, already handled by load
            pass
        else:
            # Make sure there's room for v in red
            if len(red) >= B:
                # Evict something that's not a parent of v and not v
                candidates = [x for x in red if x not in set(par) and x != v]
                if not candidates:
                    candidates = list(red)
                evict = min(candidates, key=lambda x: (remaining_uses.get(x, 0), x))
                ops.append(['delete', evict])
                red.discard(evict)
            ops.append(['compute', v])
            red.add(v)
        
        # Update remaining_uses for parents of v (they've been used once more)
        for p in par:
            remaining_uses[p] = max(0, remaining_uses.get(p, 0) - 1)
        
        # Decide what to do with v: keep red, store, or delete
        # If v is an output, it must eventually be blue (stored)
        # If v is not an output and not needed later, we can delete it
        if v in outputs:
            # Must store at the end. For now, keep red if space, else store
            if len(red) > B:
                # Shouldn't happen, but store
                ops.append(['store', v])
                stored.add(v)
                red.discard(v)
            else:
                # Keep red for now; will store at the end
                pass
        else:
            # If v is not needed by any remaining children, we can store or delete
            if remaining_uses.get(v, 0) == 0:
                # Not needed anymore. Store it so it can be loaded if somehow needed, or delete.
                # Storing costs a transfer but allows loading later. Deleting is free but forces recompute.
                # Since it's not needed, just delete to free space.
                if len(red) > B:
                    ops.append(['delete', v])
                    red.discard(v)
                else:
                    # Keep red for now, or store to free space
                    # If fast memory is tight, store it
                    if len(red) >= B:
                        ops.append(['store', v])
                        stored.add(v)
                        red.discard(v)
            else:
                # Needed later. Keep red if space, else store
                if len(red) > B:
                    ops.append(['store', v])
                    stored.add(v)
                    red.discard(v)
                # else keep red
    
    # At the end, all outputs must be blue. Store any outputs still in red.
    for v in outputs:
        if v in red:
            ops.append(['store', v])
            stored.add(v)
            red.discard(v)
    
    # Limit operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


def _recompute(v, red, stored, ops, B, inputs, parents, children, remaining_uses, vertices):
    """Recompute vertex v, ensuring its parents are red. v must not be an input."""
    if v in red:
        return
    par = parents.get(v, [])
    for p in par:
        if p in red:
            continue
        if len(red) >= B:
            candidates = [x for x in red if x not in set(par) and x != v]
            if not candidates:
                candidates = list(red)
            evict = min(candidates, key=lambda x: (remaining_uses.get(x, 0), x))
            ops.append(['delete', evict])
            red.discard(evict)
        if p in inputs:
            ops.append(['load', p])
            red.add(p)
        else:
            if p in stored:
                ops.append(['load', p])
                red.add(p)
            else:
                _recompute(p, red, stored, ops, B, inputs, parents, children, remaining_uses, vertices)
    if len(red) >= B:
        candidates = [x for x in red if x not in set(par) and x != v]
        if not candidates:
            candidates = list(red)
        evict = min(candidates, key=lambda x: (remaining_uses.get(x, 0), x))
        ops.append(['delete', evict])
        red.discard(evict)
    ops.append(['compute', v])
    red.add(v)

# EVOLVE-BLOCK-END
