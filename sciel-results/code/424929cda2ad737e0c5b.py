"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Precompute next-use positions for each vertex
    # next_use[v] = list of indices in needed_vertices where v is used as a parent
    next_use = defaultdict(list)
    for i, v in enumerate(needed_vertices):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                next_use[p].append(i)

    # State
    in_fast = set()
    in_slow = set(inputs)  # Inputs start in slow memory
    computed = set()
    operations = []
    
    # Track current position for FNU
    current_pos = 0

    def get_next_use(u, pos):
        """Get the next use position of u after position pos, or infinity if never used again."""
        uses = next_use.get(u, [])
        # Binary search for first use > pos
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uses):
            return uses[lo]
        return float('inf')

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    def evict_one(protect_set):
        """Evict one vertex from fast memory.
        Priority 1: Vertices not needed again (delete without store).
        Priority 2: FNU strategy for others.
        """
        if not in_fast:
            return None
        
        # Priority 1: Find a vertex that is not needed again
        # If it's in slow, delete is safe. If not in slow, delete is safe if not needed.
        # We want to free space.
        candidate_delete = None
        for u in list(in_fast):
            if u in protect_set:
                continue
            nu = get_next_use(u, current_pos)
            if nu == float('inf'):
                candidate_delete = u
                break
        
        if candidate_delete is not None:
            # We can just delete it, no need to store if not needed
            # But wait, if it's not in in_slow, it's a computed value.
            # If we delete it, it's gone. That's fine if not needed.
            delete(candidate_delete)
            return candidate_delete

        # Priority 2: FNU on the rest
        best_evict = None
        best_next_use = -1
        
        for u in list(in_fast):
            if u in protect_set:
                continue
            nu = get_next_use(u, current_pos)
            if nu > best_next_use:
                best_next_use = nu
                best_evict = u
        
        if best_evict is None:
            # All protected, pick any
            for u in in_fast:
                best_evict = u
                break
        
        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
        
        return best_evict

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        current_pos = vertex_to_idx.get(v, current_pos)
        
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict: Find best candidate to remove using FNU
                # Protect current parents that we still need to load
                remaining_parents = set(parents) - {p}
                evict_one(remaining_parents | {v})
            
            load(p)

        # Compute v
        if v not in in_fast:
            compute(v)
        else:
            # v was already in fast? Maybe from a previous computation?
            if v in inputs:
                pass # It's an input, already in fast (loaded above)
            else:
                # This case shouldn't happen in a standard topo pass
                pass
        
        # Optimization: If v is not an output and is not needed in the future, delete it immediately
        # This frees up fast memory for subsequent vertices
        if v not in outputs:
            nu_v = get_next_use(v, current_pos)
            if nu_v == float('inf'):
                delete(v)

    # Ensure all outputs are in fast memory at the end
    # First, evict non-outputs to make room if needed
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something that's not an output
                evict_one(outputs)
            
            load(v)

    # Final check: ensure all outputs are in fast
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) < fast_memory_budget:
                load(v)
            else:
                # Emergency: evict anything
                if in_fast:
                    u = next(iter(in_fast))
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)
                    load(v)

    return operations
# EVOLVE-BLOCK-END
