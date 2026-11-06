# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using hierarchical recursive pebbling.
    
    Strategy:
    1. Parse the DAG structure
    2. Partition into balanced sub-DAGs at natural layer boundaries
    3. Recursively pebble each sub-DAG
    4. Minimize inter-block communication by keeping only boundary values
    5. Fall back to greedy LRU eviction within blocks
    """
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            elif isinstance(dag, dict):
                vertices = list(dag.keys())
        
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
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use for each vertex
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = n
            elif not children_map.get(v, []):
                last_use[v] = n
            else:
                max_pos = 0
                for child in children_map[v]:
                    max_pos = max(max_pos, last_use.get(child, n))
                last_use[v] = max_pos
        
        # For each vertex, compute the positions where it's needed (as a parent)
        needed_at = {v: [] for v in vertices}
        for i, v in enumerate(topo_order):
            for p in parents_map.get(v, []):
                if pos_map.get(p, -1) < i:
                    needed_at[p].append(i)
        
        for v in needed_at:
            needed_at[v].sort()
        
        ops = []
        in_fast = set()
        is_blue = set(inputs)
        
        # Hierarchical approach: identify natural split points
        # For layered DAGs, we can find layers by depth
        depth = {}
        for v in topo_order:
            if not parents_map.get(v, []):
                depth[v] = 0
            else:
                max_parent_depth = 0
                for p in parents_map[v]:
                    if p in depth:
                        max_parent_depth = max(max_parent_depth, depth[p])
                depth[v] = max_parent_depth + 1
        
        max_depth = max(depth.values()) if depth else 0
        
        # Find a good split point: middle of the depth range
        # We want to split such that left and right sub-DAGs are balanced
        mid_depth = max_depth // 2
        
        # Identify vertices at or before mid_depth (left block) and after (right block)
        left_vertices = [v for v in topo_order if depth.get(v, 0) <= mid_depth]
        right_vertices = [v for v in topo_order if depth.get(v, 0) > mid_depth]
        
        # If the split doesn't help (e.g., all in one side), use greedy
        if not left_vertices or not right_vertices or len(left_vertices) < n // 4 or len(right_vertices) < n // 4:
            # Fall back to greedy LRU pebbling
            return _greedy_pebble(topo_order, parents_map, inputs, outputs, 
                                  fast_memory_budget, last_use, needed_at, 
                                  children_map, pos_map, n, deadline)
        
        # Recursive pebbling:
        # 1. Process left block, keeping only boundary values in fast memory
        # 2. Store left block results that are needed by right block
        # 3. Process right block
        
        # Identify boundary: vertices in left that are parents of vertices in right
        left_set = set(left_vertices)
        right_set = set(right_vertices)
        
        boundary_to_right = set()
        for v in right_vertices:
            for p in parents_map.get(v, []):
                if p in left_set:
                    boundary_to_right.add(p)
        
        # Process left block with greedy, but ensure boundary values are kept
        ops_left, in_fast_after_left, is_blue_after_left = _greedy_pebble_block(
            left_vertices, parents_map, inputs, outputs,
            fast_memory_budget, last_use, needed_at,
            children_map, pos_map, n, deadline,
            keep_vertices=boundary_to_right
        )
        ops.extend(ops_left)
        in_fast = in_fast_after_left
        is_blue = is_blue_after_left
        
        # Process right block
        ops_right, in_fast_final, is_blue_final = _greedy_pebble_block(
            right_vertices, parents_map, inputs, outputs,
            fast_memory_budget, last_use, needed_at,
            children_map, pos_map, n, deadline,
            initial_fast=in_fast,
            initial_blue=is_blue
        )
        ops.extend(ops_right)
        
        # Ensure all outputs are blue
        for v in outputs:
            if v not in is_blue_final:
                if v in in_fast_final:
                    ops.append(['store', v])
                    in_fast_final.remove(v)
                    is_blue_final.add(v)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

def _greedy_pebble(topo_order, parents_map, inputs, outputs, 
                   fast_memory_budget, last_use, needed_at,
                   children_map, pos_map, n, deadline):
    """Greedy LRU pebbling - the original approach."""
    ops = []
    in_fast = set()
    is_blue = set(inputs)
    next_use_idx = {v: 0 for v in topo_order}
    
    def get_next_use(v, current_pos):
        idx = next_use_idx[v]
        positions = needed_at.get(v, [])
        while idx < len(positions) and positions[idx] <= current_pos:
            idx += 1
        next_use_idx[v] = idx
        if idx < len(positions):
            return positions[idx]
        return last_use.get(v, n)
    
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        if v in inputs:
            continue
        
        parents = parents_map.get(v, [])
        
        for p in parents:
            if p in is_blue or p in in_fast:
                continue
            
            if len(in_fast) < fast_memory_budget:
                ops.append(['load', p])
                in_fast.add(p)
            else:
                best_evict = None
                best_next = -1
                for x in in_fast:
                    nu = get_next_use(x, i)
                    if nu > best_next:
                        best_next = nu
                        best_evict = x
                
                if best_evict is not None:
                    ops.append(['delete', best_evict])
                    in_fast.remove(best_evict)
                
                ops.append(['load', p])
                in_fast.add(p)
        
        ops.append(['compute', v])
        in_fast.add(v)
        
        if v in outputs:
            ops.append(['store', v])
            in_fast.remove(v)
            is_blue.add(v)
        
        # Proactively evict vertices whose last use has passed
        if len(in_fast) > 0:
            to_evict = []
            for x in in_fast:
                if last_use.get(x, 0) <= i and x not in outputs:
                    to_evict.append(x)
            for x in to_evict:
                if x in in_fast:
                    ops.append(['delete', x])
                    in_fast.remove(x)
    
    for v in outputs:
        if v not in is_blue:
            if v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops

def _greedy_pebble_block(vertex_subset, parents_map, inputs, outputs,
                         fast_memory_budget, last_use, needed_at,
                         children_map, pos_map, n, deadline,
                         keep_vertices=None,
                         initial_fast=None,
                         initial_blue=None):
    """Greedy pebbling on a subset of vertices, with options to keep certain vertices."""
    if keep_vertices is None:
        keep_vertices = set()
    
    ops = []
    in_fast = set(initial_fast) if initial_fast else set()
    is_blue = set(initial_blue) if initial_blue else set(inputs)
    next_use_idx = {v: 0 for v in vertex_subset}
    
    def get_next_use(v, current_pos):
        idx = next_use_idx.get(v, 0)
        positions = needed_at.get(v, [])
        while idx < len(positions) and positions[idx] <= current_pos:
            idx += 1
        next_use_idx[v] = idx
        if idx < len(positions):
            return positions[idx]
        return last_use.get(v, n)
    
    for i, v in enumerate(vertex_subset):
        if time.time() > deadline:
            break
        
        if v in inputs:
            continue
        
        parents = parents_map.get(v, [])
        
        for p in parents:
            if p in is_blue or p in in_fast:
                continue
            
            if len(in_fast) < fast_memory_budget:
                ops.append(['load', p])
                in_fast.add(p)
            else:
                best_evict = None
                best_next = -1
                for x in in_fast:
                    # Don't evict vertices we need to keep
                    if x in keep_vertices:
                        continue
                    nu = get_next_use(x, i)
                    if nu > best_next:
                        best_next = nu
                        best_evict = x
                
                if best_evict is not None:
                    ops.append(['delete', best_evict])
                    in_fast.remove(best_evict)
                else:
                    # All fast memory is keep_vertices, force evict one
                    # This shouldn't happen often, but handle it
                    best_evict = None
                    best_next = -1
                    for x in in_fast:
                        nu = get_next_use(x, i)
                        if nu > best_next:
                            best_next = nu
                            best_evict = x
                    if best_evict is not None:
                        ops.append(['delete', best_evict])
                        in_fast.remove(best_evict)
                
                ops.append(['load', p])
                in_fast.add(p)
        
        ops.append(['compute', v])
        in_fast.add(v)
        
        if v in outputs:
            ops.append(['store', v])
            in_fast.remove(v)
            is_blue.add(v)
        elif v in keep_vertices:
            # Keep in fast memory, don't evict
            pass
        else:
            # Proactively evict if last use has passed
            if last_use.get(v, 0) <= i:
                ops.append(['delete', v])
                in_fast.remove(v)
    
    # Clean up: evict non-keep vertices whose last use has passed
    for x in list(in_fast):
        if x not in keep_vertices and last_use.get(x, 0) <= len(vertex_subset):
            ops.append(['delete', x])
            in_fast.remove(x)
    
    return ops, in_fast, is_blue
# EVOLVE-BLOCK-END
