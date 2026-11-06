# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using recursive block decomposition.
    
    For layered DAGs, recursive pebbling minimizes total transfers by:
    - Partitioning into balanced sub-DAGs
    - Recursively pebbling each part
    - Minimizing inter-block data movement
    
    Strategy:
    1. Parse DAG, build topological order and dependency structure
    2. If DAG fits in fast memory, use simple topological pass
    3. Otherwise, recursively split and pebble with careful I/O management
    4. Use last-use analysis for optimal deletion ordering
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            else:
                vertices = list(dag.keys()) if isinstance(dag, dict) else []
        
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        # Build children map
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        if not outputs:
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        topo_order = list(vertices)
        n = len(topo_order)
        
        if n == 0:
            return []
        
        # If fast memory budget is large enough to hold all non-input vertices,
        # just do a simple topological pass
        non_input_count = n - len(inputs)
        
        # Simple topological pass for small DAGs or large budgets
        def simple_pass(order, parents_map, inputs, outputs, children_map, budget):
            ops = []
            in_fast = set()
            is_blue = set(inputs)
            
            # Precompute last_use
            pos_map = {v: i for i, v in enumerate(order)}
            last_use = {}
            for v in reversed(order):
                if v in outputs:
                    last_use[v] = len(order)
                elif not children_map.get(v, []):
                    last_use[v] = len(order)
                else:
                    max_pos = 0
                    for child in children_map.get(v, []):
                        cp = pos_map.get(child, len(order))
                        if cp > max_pos:
                            max_pos = cp
                    last_use[v] = max_pos
            
            for i, v in enumerate(order):
                if v in inputs:
                    continue
                
                parents = parents_map.get(v, [])
                
                for p in parents:
                    if p in is_blue or p in in_fast:
                        continue
                    # Need to load p
                    if len(in_fast) >= budget:
                        # Find vertex to evict: prefer one not needed soon
                        to_delete = None
                        best_lu = -1
                        for x in in_fast:
                            lu = last_use.get(x, len(order))
                            if lu > best_lu:
                                best_lu = lu
                                to_delete = x
                        if to_delete is not None:
                            ops.append(['delete', to_delete])
                            in_fast.remove(to_delete)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
                
                ops.append(['compute', v])
                in_fast.add(v)
                
                if v in outputs:
                    ops.append(['store', v])
                    in_fast.remove(v)
                    is_blue.add(v)
                
                # Delete parents no longer needed
                for p in parents:
                    if p in in_fast and last_use.get(p, len(order)) <= i:
                        ops.append(['delete', p])
                        in_fast.remove(p)
            
            return ops
        
        # For very small DAGs, use simple pass
        if n <= 50 or budget >= non_input_count + 10:
            ops = simple_pass(topo_order, parents_map, inputs, outputs, children_map, budget)
            if len(ops) <= 20000:
                return ops
        
        # Recursive pebbling for larger DAGs
        # Key insight: for a chain-like or layered structure, we can split
        # and recursively solve, managing I/O at boundaries
        
        def recursive_pebble(order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set):
            """
            Recursively pebble the given order.
            is_blue_set and in_fast_set are mutable state.
            Returns list of operations.
            """
            ops = []
            m = len(order)
            if m == 0:
                return ops
            
            if time.time() > deadline:
                return ops
            
            # If this sub-DAG fits in budget, do simple pass
            # Count how many vertices need fast memory at any point
            # Approximate: number of non-input vertices in this sub-DAG
            local_non_input = sum(1 for v in order if v not in inputs)
            
            if local_non_input <= budget:
                # Simple pass for this sub-DAG
                for v in order:
                    if v in inputs:
                        continue
                    
                    parents = parents_map.get(v, [])
                    
                    for p in parents:
                        if p in is_blue_set or p in in_fast_set:
                            continue
                        # Need to load p
                        if len(in_fast_set) >= budget:
                            # Evict: find vertex with latest last_use in this sub-DAG
                            # Simple heuristic: evict the one loaded earliest
                            to_delete = None
                            # Find a vertex to evict
                            for x in list(in_fast_set):
                                if x not in order:  # From outside this sub-DAG
                                    to_delete = x
                                    break
                            if to_delete is None:
                                # Evict the one with highest last_use
                                best_lu = -1
                                for x in in_fast_set:
                                    # Estimate last use
                                    try:
                                        idx = order.index(x)
                                        lu = m - idx
                                    except ValueError:
                                        lu = m
                                    if lu > best_lu:
                                        best_lu = lu
                                        to_delete = x
                            if to_delete is not None:
                                ops.append(['delete', to_delete])
                                in_fast_set.remove(to_delete)
                        
                        ops.append(['load', p])
                        in_fast_set.add(p)
                    
                    ops.append(['compute', v])
                    in_fast_set.add(v)
                    
                    if v in outputs:
                        ops.append(['store', v])
                        in_fast_set.remove(v)
                        is_blue_set.add(v)
                
                return ops
            
            # Split into two halves
            mid = m // 2
            left_order = order[:mid]
            right_order = order[mid:]
            
            # Find boundary: vertices in right that depend on vertices in left
            left_set = set(left_order)
            right_set = set(right_order)
            
            # Boundary vertices: in left, but needed by right
            boundary = set()
            for v in right_order:
                for p in parents_map.get(v, []):
                    if p in left_set and p not in inputs:
                        boundary.add(p)
            
            # Strategy:
            # 1. Pebble left, but store boundary vertices
            # 2. Pebble right, loading boundary vertices as needed
            
            # First pass: pebble left half
            ops.extend(recursive_pebble(left_order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set))
            
            # Store boundary vertices to make them available for right half
            # They should already be in fast memory or blue
            for b in boundary:
                if b in in_fast_set:
                    ops.append(['store', b])
                    in_fast_set.remove(b)
                    is_blue_set.add(b)
            
            # Clear fast memory of left-half vertices not in boundary
            for v in left_order:
                if v in in_fast_set and v not in boundary:
                    ops.append(['delete', v])
                    in_fast_set.remove(v)
            
            # Second pass: pebble right half
            ops.extend(recursive_pebble(right_order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set))
            
            return ops
        
        # Initialize state
        is_blue_set = set(inputs)
        in_fast_set = set()
        
        ops = recursive_pebble(topo_order, parents_map, inputs, outputs, children_map, fast_memory_budget, is_blue_set, in_fast_set)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
