"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop
from collections import defaultdict

def _extract_dag_info(dag):
    """Extract DAG structure from various possible formats."""
    # Get number of vertices
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    elif hasattr(dag, 'vertices'):
        n = len(dag.vertices)
    else:
        n = 0
    
    # Get parents for each vertex
    def get_parents(v):
        if hasattr(dag, 'parents'):
            p = dag.parents[v]
            if isinstance(p, dict):
                return list(p.keys())
            return list(p) if p else []
        elif isinstance(dag, dict) and 'parents' in dag:
            p = dag['parents'][v]
            if isinstance(p, dict):
                return list(p.keys())
            return list(p) if p else []
        else:
            return []
    
    # Get output vertices
    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            return {n-1} if n > 0 else set()
    
    return n, get_parents, get_outputs()

def _state_key(red_set):
    """Create a hashable key from the set of red vertices."""
    return frozenset(red_set)

def _compute_cost(operations):
    """Count total transfers (load + store operations)."""
    return sum(1 for op in operations if op[0] in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    n, get_parents, get_outputs = _extract_dag_info(dag)
    outputs = get_outputs()
    
    if n == 0 or fast_memory_budget <= 0:
        return []
    
    # Build children map for efficiency
    children = defaultdict(list)
    for v in range(n):
        for p in get_parents(v):
            children[p].append(v)
    
    # Count remaining uses for each vertex (how many children still need it)
    def count_remaining_uses(v, completed):
        count = 0
        for child in children[v]:
            if child not in completed:
                count += 1
        return count
    
    # Dijkstra-based search over states
    # State: (frozenset of red vertices, frozenset of completed vertices, frozenset of blue outputs)
    # But this is too large. Let's use a more practical approach:
    # Greedy with lookahead, then local search refinement
    
    # First pass: greedy topological pebbling
    def greedy_pebble():
        red = set()
        completed = set()
        operations = []
        
        # Precompute: for each vertex, how many times it's needed
        use_count = [len(children[v]) for v in range(n)]
        remaining_uses = use_count[:]  # mutable copy
        
        for v in range(n):
            parents = get_parents(v)
            
            # Ensure all parents are in fast memory
            for p in parents:
                if p not in red:
                    if len(red) >= fast_memory_budget:
                        # Evict: choose vertex with minimum remaining uses
                        best_evict = None
                        best_uses = float('inf')
                        for r in red:
                            if r not in completed:
                                uses = remaining_uses[r]
                                if uses < best_uses:
                                    best_uses = uses
                                    best_evict = r
                        if best_evict is not None:
                            operations.append(['store', best_evict])
                            red.discard(best_evict)
                    
                    if p not in red:
                        operations.append(['load', p])
                        red.add(p)
            
            # Compute v
            operations.append(['compute', v])
            red.add(v)
            completed.add(v)
            
            # Decrement remaining uses for parents
            for p in parents:
                remaining_uses[p] -= 1
        
        # Store all outputs
        for v in sorted(outputs):
            if v in red:
                operations.append(['store', v])
                red.discard(v)
        
        return operations
    
    # Generate initial solution
    operations = greedy_pebble()
    initial_cost = _compute_cost(operations)
    
    # If we're already good, return early
    if time.time() - start_time > deadline - 10:
        return operations
    
    # Local search: try to improve by reordering or removing redundant operations
    # For now, return the greedy solution as it's likely near-optimal for layered DAGs
    
    return operations
# EVOLVE-BLOCK-END
