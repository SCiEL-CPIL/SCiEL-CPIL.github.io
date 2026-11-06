"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with memory-aware scheduling and checkpoint optimization.
    """
    start_time = time.time()
    DEADLINE = start_time + 55  # Safety margin before 60s kill
    
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
            parents = dict(dag.parents) if not isinstance(dag.parents, dict) else dag.parents
        elif isinstance(dag, dict) and 'parents' in dag:
            parents = dag['parents']
        else:
            parents = {}
        
        # Normalize parents to dict: vertex -> list of parents
        if not isinstance(parents, dict):
            if isinstance(parents, (list, tuple)):
                parents = {v: list(parents[i]) for i, v in enumerate(vertices)}
            else:
                parents = {v: [] for v in vertices}
        
        # Ensure all vertices have entries
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
        
        # If budget is large enough, do simple topological pass
        if fast_memory_budget >= n:
            ops = []
            in_fast = set()
            
            # Load all inputs
            for v in sorted(inputs):
                ops.append(['load', v])
                in_fast.add(v)
            
            # Compute in topological order
            in_degree = {v: len(parents[v]) for v in vertices}
            queue = deque([v for v in vertices if in_degree[v] == 0])
            topo_order = []
            while queue:
                v = queue.popleft()
                topo_order.append(v)
                for w in vertices:
                    if v in parents[w]:
                        in_degree[w] -= 1
                        if in_degree[w] == 0:
                            queue.append(w)
            
            # Precompute last use for each vertex
            last_use = {v: -1 for v in vertices}
            for i, v in enumerate(topo_order):
                for p in parents[v]:
                    last_use[p] = i
            
            for i, v in enumerate(topo_order):
                if v in inputs:
                    continue
                # Check all parents are in fast
                for p in parents[v]:
                    if p not in in_fast:
                        ops.append(['load', p])
                        in_fast.add(p)
                ops.append(['compute', v])
                in_fast.add(v)
                
                # Delete vertices that are no longer needed
                for p in list(in_fast):
                    if p != v and last_use.get(p, -1) < i and p not in outputs:
                        ops.append(['delete', p])
                        in_fast.remove(p)
            
            # Store outputs
            for v in sorted(outputs):
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.remove(v)
            
            return ops
        
        # Memory-constrained case: use layer-based approach with optimal checkpointing
        # Compute depth of each vertex
        depth = {}
        for v in vertices:
            if not parents[v]:
                depth[v] = 0
            else:
                depth[v] = max(depth.get(p, 0) + 1 for p in parents[v])
        
        # Group by depth
        max_depth = max(depth.values()) if depth else 0
        layers = defaultdict(list)
        for v in vertices:
            layers[depth[v]].append(v)
        
        # For each layer, we need to decide which vertices to keep in fast memory
        # and which to recompute. We use a greedy approach based on the cost of
        # recomputation vs. the memory saved.
        
        # First, let's understand the structure better
        # For a layered DAG, each vertex in layer d depends on vertices in layer d-1
        # We can process layer by layer, keeping only what's needed
        
        ops = []
        in_fast = set()
        
        # Precompute for each vertex, which future layers need it
        # next_use_layer[v] = the next layer index where v is needed as a parent
        next_use_layer = {}
        for d in range(max_depth + 1):
            for v in layers[d]:
                for p in parents[v]:
                    if p not in next_use_layer or d < next_use_layer[p]:
                        next_use_layer[p] = d
        
        # Process layer by layer
        for d in range(max_depth + 1):
            if time.time() > DEADLINE:
                break
            
            layer_vertices = sorted(layers.get(d, []))
            
            # Load any parents from previous layers that are needed but not in fast
            for v in layer_vertices:
                for p in parents[v]:
                    if p not in in_fast:
                        # Check if p is an input or intermediate
                        # For now, load it (this may cause memory overflow, but we'll handle eviction)
                        if len(in_fast) >= fast_memory_budget:
                            # Evict the vertex with the farthest next use
                            # Find the vertex in in_fast with the maximum next_use_layer
                            evict_v = None
                            max_next_use = -1
                            for x in in_fast:
                                nu = next_use_layer.get(x, max_depth + 1)
                                if nu > max_next_use:
                                    max_next_use = nu
                                    evict_v = x
                            if evict_v is not None:
                                ops.append(['store', evict_v])
                                in_fast.remove(evict_v)
                        
                        ops.append(['load', p])
                        in_fast.add(p)
            
            # Compute vertices in this layer
            for v in layer_vertices:
                if v in inputs:
                    continue
                # Verify parents are in fast
                for p in parents[v]:
                    if p not in in_fast:
                        # This shouldn't happen if we loaded correctly, but just in case
                        if len(in_fast) >= fast_memory_budget:
                            evict_v = None
                            max_next_use = -1
                            for x in in_fast:
                                nu = next_use_layer.get(x, max_depth + 1)
                                if nu > max_next_use:
                                    max_next_use = nu
                                    evict_v = x
                            if evict_v is not None:
                                ops.append(['store', evict_v])
                                in_fast.remove(evict_v)
                        ops.append(['load', p])
                        in_fast.add(p)
                ops.append(['compute', v])
                in_fast.add(v)
            
            # After computing this layer, clean up
            # Delete vertices from previous layers that are no longer needed
            # and are not outputs
            if d > 0:
                prev_layer = layers.get(d - 1, [])
                for v in prev_layer:
                    if v in in_fast and v not in outputs:
                        # Check if any future vertex needs v as parent
                        nu = next_use_layer.get(v, max_depth + 1)
                        if nu <= d:
                            ops.append(['delete', v])
                            in_fast.remove(v)
            
            # If this layer contains outputs, store them
            for v in layer_vertices:
                if v in outputs and v in in_fast:
                    ops.append(['store', v])
                    in_fast.remove(v)
        
        # Final cleanup: store any remaining outputs
        for v in sorted(outputs):
            if v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
        
        # Verify we have at most 20000 operations
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
    
    except Exception as e:
        # Fall back to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

# EVOLVE-BLOCK-END
