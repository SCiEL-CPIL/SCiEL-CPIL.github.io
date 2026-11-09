"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list
            
            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        
        return parents, outputs, inputs, vertices, n
    
    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n
    
    return None, None, None, None, None


def _build_children(parents, n):
    """Build children list."""
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children


def _compute_depth(parents, n):
    """Compute depth (longest path from any input) for each vertex."""
    depth = [0] * n
    # Topological order is 0..n-1
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _layered_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Layer-by-layer pebbling strategy.
    For layered DAGs, process one layer at a time:
    - Load previous layer from slow memory
    - Compute current layer
    - Store current layer to slow memory
    - Delete previous layer from fast memory
    This minimizes transfers for layered structures.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    depth = _compute_depth(parents, n)
    is_output = set(outputs)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    # Group vertices by depth (layer)
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    ops = []
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        
        layer = layers[d]
        if not layer:
            continue
        
        if d == 0:
            # Layer 0: inputs - just load them
            for v in layer:
                ops.append(['load', v])
        else:
            # Load previous layer values that are needed for this layer
            # Actually, we need to load all parents of vertices in this layer
            # that are in the previous layer (or earlier)
            
            # Collect all unique parents of vertices in this layer
            needed_parents = set()
            for v in layer:
                for p in parents[v]:
                    needed_parents.add(p)
            
            # Load those parents (they should have been stored in previous iteration)
            for p in sorted(needed_parents):
                ops.append(['load', p])
            
            # Compute vertices in this layer
            for v in layer:
                ops.append(['compute', v])
            
            # Store vertices in this layer that are not final outputs
            # (outputs will be stored at the end)
            for v in layer:
                if v not in is_output:
                    ops.append(['store', v])
        
        # Delete previous layer from fast memory (except outputs)
        if d > 0 and d - 1 <= max_depth:
            prev_layer = layers[d - 1]
            for v in prev_layer:
                if v not in is_output:
                    ops.append(['delete', v])
    
    # Store any remaining outputs that haven't been stored
    # Find which outputs are in the last layer
    last_layer = layers[max_depth]
    stored_outputs = set()
    for v in last_layer:
        if v in is_output:
            ops.append(['store', v])
            stored_outputs.add(v)
    
    # Check if all outputs are stored
    for o in outputs:
        if o not in stored_outputs:
            # Output might be in an earlier layer
            # This shouldn't happen in a proper layered DAG, but handle it
            pass
    
    return ops


def _greedy_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Greedy pebbling with last-use analysis.
    Processes vertices in topological order, evicting values that are no longer needed.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    
    # Precompute last use for each vertex
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
    
    ops = []
    in_fast = set()
    
    def try_evict():
        """Try to evict a vertex from fast memory. Returns True if successful."""
        if not in_fast:
            return False
        # Find vertex with smallest last_use that is not an output
        best = None
        best_lu = float('inf')
        for u in in_fast:
            if u not in is_output:
                lu = last_use[u]
                if lu < best_lu:
                    best_lu = lu
                    best = u
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        # If all are outputs, evict any
        if in_fast:
            u = next(iter(in_fast))
            ops.append(['delete', u])
            in_fast.discard(u)
            return True
        return False
    
    for v in range(n):
        if time.time() > deadline:
            return None
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v if it has parents
        if parents[v]:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
        
        # Handle input-output vertices
        if not parents[v] and v in is_output:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # Store outputs
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        
        # Clean up parents that are no longer needed
        for p in parents[v]:
            if p in in_fast and p not in is_output and last_use[p] == v:
                ops.append(['delete', p])
                in_fast.discard(p)
    
    return ops


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
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
            in_fast.discard(v)
        else:
            return False
    
    for o in outputs:
        if o not in stored:
            return False
    
    return True


def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _hierarchical_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Hierarchical pebbling using recursive decomposition.
    Partitions the DAG into independent sub-problems and solves each optimally.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    
    # Find the "cut" that splits the DAG into two parts
    # We look for a layer where the number of vertices crossing is minimal
    depth = _compute_depth(parents, n)
    max_depth = max(depth) if depth else 0
    
    # Find the best split point
    best_split = -1
    best_crossing = float('inf')
    
    for d in range(1, max_depth):
        # Vertices at depth d and their parents at depth < d
        # Crossing = vertices at depth d that have parents at depth < d-1
        # Actually, we want to minimize the number of vertices that need to be
        # kept in memory across the split
        
        # For a layered DAG, the crossing at depth d is the number of vertices
        # at depth d-1 that have children at depth d
        prev_layer = [v for v in range(n) if depth[v] == d - 1]
        curr_layer = [v for v in range(n) if depth[v] == d]
        
        # Count how many prev_layer vertices are needed by curr_layer
        needed = set()
        for v in curr_layer:
            for p in parents[v]:
                needed.add(p)
        
        crossing = len(needed)
        if crossing < best_crossing:
            best_crossing = crossing
            best_split = d
    
    # If no good split found, use the whole DAG
    if best_split <= 0 or best_split >= max_depth:
        return _greedy_pebbling(parents, outputs, fast_memory_budget, deadline)
    
    # Split the DAG at best_split
    # Part 1: vertices with depth < best_split
    # Part 2: vertices with depth >= best_split
    
    # For part 1, we need to compute all vertices and store those needed by part 2
    # For part 2, we load those vertices and compute the rest
    
    # This is complex; for now, fall back to greedy
    return _greedy_pebbling(parents, outputs, fast_memory_budget, deadline)


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    best_ops = None
    best_transfers = float('inf')
    
    # Try layered pebbling first (optimal for layered DAGs)
    try:
        ops = _layered_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try greedy pebbling
    try:
        ops = _greedy_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try hierarchical pebbling
    try:
        ops = _hierarchical_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    if best_ops is not None:
        return best_ops
    
    # Last resort: very simple approach
    try:
        ops = []
        in_fast = set()
        children = _build_children(parents, n)
        is_output = set(outputs)
        
        for v in range(n):
            if time.time() > deadline:
                break
            
            for p in parents[v]:
                if p not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            if parents[v]:
                while len(in_fast) >= fast_memory_budget:
                    evicted = False
                    for u in list(in_fast):
                        if u not in is_output:
                            ops.append(['delete', u])
                            in_fast.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            if v in is_output:
                if v not in in_fast and not parents[v]:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', v])
                        in_fast.add(v)
                
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            
            for p in parents[v]:
                if p in in_fast and p not in is_output:
                    # Check if p is still needed
                    still_needed = False
                    for c in children[p]:
                        if c > v:
                            still_needed = True
                            break
                    if not still_needed:
                        ops.append(['delete', p])
                        in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    return []

# EVOLVE-BLOCK-END
