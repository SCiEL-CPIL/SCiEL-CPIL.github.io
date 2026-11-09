# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

_DEADLINE = 55.0  # Safety margin before 60s limit

def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return parents.get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    else:
        return []

def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        # Try to infer from parents
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []

def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs

def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {}
    children = defaultdict(list)
    
    for v in all_v:
        ps = _get_parents(dag, v)
        parents[v] = ps
        for p in ps:
            children[p].append(v)
    
    # Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = deque([v for v in all_v if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        # Cycle or error, just return all vertices
        return all_v
    
    return topo

def _pebble_greedy(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with reference counting and eviction.
    """
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = set(_get_inputs(dag))
    outputs = set(_get_outputs(dag))
    
    # Precompute children and reference counts
    children = defaultdict(list)
    ref_count = defaultdict(int)
    for v in all_v:
        ps = _get_parents(dag, v)
        for p in ps:
            children[p].append(v)
            ref_count[p] += 1
    
    # Outputs are "used" once at the end
    for o in outputs:
        ref_count[o] += 1 # Ensure outputs are kept until stored? 
        # Actually, if an output has no children, ref_count is 0 from parents.
        # We need to keep it until we store it.
        # We'll handle outputs specifically.
        
    # Reset ref_count for logic: ref_count[v] = number of children that need v.
    # If ref_count[v] == 0, v is only needed if it's an output (to be stored) or already used.
    
    operations = []
    in_fast = set()
    # next_use[v] = index in topo order of the next child that needs v
    next_use = {}
    
    # Precompute next_use for each vertex
    # For each v, look at its children in topo order.
    for v in all_v:
        ch = children[v]
        if ch:
            # Find the earliest child in topo order
            # Since topo is a list, we can find the min index
            min_idx = min(topo.index(c) for c in ch)
            next_use[v] = min_idx
        else:
            next_use[v] = len(topo) # Never used again (except if output)
            
    # Special case for outputs: they must be kept until stored.
    # We will store them immediately after computation if they are the last use?
    # No, we must store them to slow memory.
    # If an output is computed, we can store it immediately if we don't need it for anything else.
    # Since it's an output, it has no children. So next_use is len(topo).
    # We can delete it from fast memory after storing.
    
    # Simulation
    for i, v in enumerate(topo):
        if time.time() - deadline[0] > _DEADLINE:
            break
            
        parents = _get_parents(dag, v)
        
        if not parents:
            # Input vertex
            # It starts blue in slow memory.
            # Do we need to load it?
            # Yes, if it has children.
            if children[v]:
                # Check if v is already in fast memory
                if v not in in_fast:
                    # Need to load v
                    if len(in_fast) >= fast_memory_budget:
                        # Evict someone
                        # Find candidate to evict: 
                        # 1. Not an output that hasn't been stored yet? (We store outputs immediately)
                        # 2. Has lowest next_use (furthest in future) or already used (next_use > i)?
                        # Actually, if next_use > i, it's already been used by all its children?
                        # No, next_use is the *first* child. If we are at i, and next_use[v] < i, it was used.
                        # We need to evict a node that is NOT needed for the current step or immediate future.
                        
                        # Heuristic: Evict node with max next_use (least urgent)
                        # Or evict node that is already "dead" (next_use < i and not output?)
                        
                        candidates = []
                        for u in in_fast:
                            # Don't evict if it's a parent of current v? 
                            # We are loading parents of v. So v's parents are safe.
                            # We are evicting to make room for a parent of v.
                            # So we can evict any u that is not a parent of v.
                            
                            # Priority: 
                            # 1. u is not a parent of v.
                            # 2. u's next_use is far in the future.
                            
                            if u not in parents:
                                nu = next_use.get(u, len(topo))
                                candidates.append((nu, u))
                        
                        if candidates:
                            # Sort by next_use descending (evict the one needed latest)
                            candidates.sort(key=lambda x: -x[0])
                            victim = candidates[0][1]
                            operations.append(['delete', victim])
                            in_fast.remove(victim)
                        
                        # If still full and no candidates, we have a problem (cycle or logic error)
                        # But in a DAG, we should always be able to evict something not needed.
                        
                    operations.append(['load', v])
                    in_fast.add(v)
            else:
                # Input with no children? Shouldn't happen in a valid DAG for pebbling unless it's an output too?
                # If it's an output, we need to load and store it.
                if v in outputs:
                    if v not in in_fast:
                        if len(in_fast) >= fast_memory_budget:
                             # Evict
                             candidates = []
                             for u in in_fast:
                                nu = next_use.get(u, len(topo))
                                candidates.append((nu, u))
                             if candidates:
                                candidates.sort(key=lambda x: -x[0])
                                victim = candidates[0][1]
                                operations.append(['delete', victim])
                                in_fast.remove(victim)
                        operations.append(['load', v])
                        in_fast.add(v)
                    operations.append(['store', v])
                    in_fast.remove(v)
            
        else:
            # Internal vertex
            # 1. Ensure all parents are in fast memory
            for p in parents:
                if p not in in_fast:
                    # Need to load p
                    if len(in_fast) >= fast_memory_budget:
                        # Evict
                        candidates = []
                        for u in in_fast:
                            if u not in parents: # Don't evict other parents of v
                                nu = next_use.get(u, len(topo))
                                candidates.append((nu, u))
                        
                        if candidates:
                            candidates.sort(key=lambda x: -x[0])
                            victim = candidates[0][1]
                            operations.append(['delete', victim])
                            in_fast.remove(victim)
                        
                    operations.append(['load', p])
                    in_fast.add(p)
            
            # 2. Compute v
            operations.append(['compute', v])
            in_fast.add(v)
            
            # 3. Update next_use for parents? 
            # We don't need to update dynamically if we precomputed "first use".
            # But if a parent has multiple children, we used it now.
            # The "next_use" precomputed is the *first* child.
            # If we are processing the first child, next_use is correct.
            # If we are processing a later child, the parent might have been evicted and re-loaded.
            # This heuristic is approximate.
            
            # 4. Check if v can be deleted/stored
            if v in outputs:
                # Store it immediately
                operations.append(['store', v])
                in_fast.remove(v)
            else:
                # If v has no more children (ref_count becomes 0), we can delete it?
                # We don't track dynamic ref_count in this loop easily without updating.
                # Let's just keep it until evicted or if it's the last one.
                # Actually, if next_use[v] < i, it was already used.
                # If next_use[v] == i, it's being used now? No, v is the current node.
                # We need to know if v will be used again.
                # If children[v] is empty, it won't be used again.
                if not children[v]:
                    # No children, not an output? Then it's a dead end. Delete it.
                    operations.append(['delete', v])
                    in_fast.remove(v)
                # If it has children, keep it for now.
    
    # Final phase: Ensure all outputs are stored
    # If some outputs were not stored (e.g. if they were evicted?), we need to load and store them.
    # But our logic stores outputs immediately upon computation.
    # So this should be safe.
    
    return operations

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    try:
        ops = _pebble_greedy(dag, fast_memory_budget, deadline)
        
        if not ops:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        # Validate
        if _validate_operations(ops, dag, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # If invalid, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
        
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
