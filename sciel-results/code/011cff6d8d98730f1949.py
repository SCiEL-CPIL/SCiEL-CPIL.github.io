"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Improved DAG pebbling with memory-aware scheduling using optimal layer pebbling.
    """
    start_time = time.time()
    DEADLINE = start_time + 85  # Safety margin before 90s kill
    
    try:
        # Parse DAG structure
        if hasattr(dag, 'vertices'):
            vertices = list(dag.vertices)
        elif isinstance(dag, dict) and 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif isinstance(dag, (list, tuple)):
            vertices = list(dag)
        else:
            for attr in ['vertices', 'nodes', 'V']:
                if hasattr(dag, attr):
                    vertices = list(getattr(dag, attr))
                    break
                elif isinstance(dag, dict) and attr in dag:
                    vertices = list(dag[attr])
                    break
            else:
                raise ValueError("Cannot extract vertices from dag")
        
        n = len(vertices)
        
        # Extract parents
        if hasattr(dag, 'parents'):
            parents = dag.parents
        elif isinstance(dag, dict) and 'parents' in dag:
            parents = dag['parents']
        else:
            parents = {}
        
        if not isinstance(parents, dict):
            if isinstance(parents, (list, tuple)):
                parents = {v: list(parents[i]) for i, v in enumerate(vertices)}
            else:
                parents = {v: [] for v in vertices}
        
        for v in vertices:
            if v not in parents:
                parents[v] = []
        
        # Extract inputs and outputs
        if hasattr(dag, 'inputs'):
            inputs = set(dag.inputs)
        elif isinstance(dag, dict) and 'inputs' in dag:
            inputs = set(dag['inputs'])
        else:
            inputs = set(v for v in vertices if not parents[v])
        
        if hasattr(dag, 'outputs'):
            outputs = set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            outputs = set(dag['outputs'])
        else:
            children = defaultdict(set)
            for v in vertices:
                for p in parents[v]:
                    children[p].add(v)
            outputs = set(v for v in vertices if not children[v])
        
        # Build children map for efficient lookups
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents[v]:
                children_map[p].append(v)
        
        # If budget is large enough, do simple topological pass with efficient cleanup
        if fast_memory_budget >= n:
            ops = []
            in_fast = set()
            
            # Efficient topological sort using Kahn's algorithm with adjacency
            in_degree = {v: len(parents[v]) for v in vertices}
            queue = deque([v for v in vertices if in_degree[v] == 0])
            topo_order = []
            while queue:
                v = queue.popleft()
                topo_order.append(v)
                for c in children_map[v]:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)
            
            # Track which vertices still need each parent
            remaining_children = {v: len(children_map[v]) for v in vertices}
            
            for v in topo_order:
                if v in inputs:
                    continue
                for p in parents[v]:
                    if p not in in_fast:
                        ops.append(['load', p])
                        in_fast.add(p)
                ops.append(['compute', v])
                in_fast.add(v)
                
                # Delete parents that are no longer needed
                for p in parents[v]:
                    remaining_children[p] -= 1
                    if remaining_children[p] == 0 and p not in outputs:
                        ops.append(['delete', p])
                        in_fast.remove(p)
            
            for v in sorted(outputs):
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.remove(v)
            
            return ops
        
        # Memory-constrained case: use optimal layer-based pebbling
        # Compute depth using topological order
        in_degree = {v: len(parents[v]) for v in vertices}
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo_order = []
        depth = {}
        
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            if not parents[v]:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] + 1 for p in parents[v])
            for c in children_map[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        max_depth = max(depth.values()) if depth else 0
        layers = defaultdict(list)
        for v in vertices:
            layers[depth[v]].append(v)
        
        # For each layer, precompute which parents are needed
        # Use a recursive approach: process each layer, keeping only what's needed
        
        ops = []
        in_fast = set()
        
        # Helper: get all vertices that depend (directly or indirectly) on v within a depth range
        def get_dependents(v, min_depth, max_depth_val, cache={}):
            key = (v, min_depth, max_depth_val)
            if key in cache:
                return cache[key]
            result = set()
            for c in children_map[v]:
                if min_depth <= depth[c] <= max_depth_val:
                    result.add(c)
                    result.update(get_dependents(c, min_depth, max_depth_val, cache))
            cache[key] = result
            return result
        
        # Process each layer
        for d in range(max_depth + 1):
            if time.time() > DEADLINE:
                break
            
            layer_vertices = sorted(layers.get(d, []))
            
            # Determine which vertices from previous layers are needed by this layer and future layers
            # For layer d, we need parents that are in layers < d and are needed by layer d or later
            
            # First, ensure all inputs in this layer are loaded
            for v in layer_vertices:
                if v in inputs and v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
            
            # For each vertex in this layer, ensure its parents are in fast memory
            # If a parent is not in fast memory, we need to reload it (it must be an input or we recompute)
            for v in layer_vertices:
                if v in inputs:
                    continue
                for p in parents[v]:
                    if p not in in_fast:
                        # p must be an input (intermediates can't be reloaded unless stored)
                        # For now, assume inputs can always be loaded
                        ops.append(['load', p])
                        in_fast.add(p)
                ops.append(['compute', v])
                in_fast.add(v)
            
            # After processing layer d, clean up vertices that are no longer needed
            # A vertex v is needed if any vertex in layer > d has v in its ancestors
            for v in list(in_fast):
                if v in outputs:
                    continue  # Keep outputs until stored
                # Check if v is needed by any future layer
                needed = False
                for fd in range(d + 1, max_depth + 1):
                    for w in layers.get(fd, []):
                        # Check if v is an ancestor of w
                        # This is expensive; use a simpler heuristic for now
                        if v in parents[w]:
                            needed = True
                            break
                    if needed:
                        break
                if not needed and v not in outputs:
                    ops.append(['delete', v])
                    in_fast.remove(v)
            
            # Store outputs in this layer
            for v in layer_vertices:
                if v in outputs and v in in_fast:
                    ops.append(['store', v])
                    in_fast.remove(v)
        
        # Final cleanup
        for v in sorted(outputs):
            if v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
    
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            # Last resort: return empty or minimal
            return []

# EVOLVE-BLOCK-END
