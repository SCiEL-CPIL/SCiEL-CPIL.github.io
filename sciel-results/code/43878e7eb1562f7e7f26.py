"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve DAG pebbling with communication minimization.
    
    Args:
        dag: dict with keys 'vertices' (list of vertex ids), 
             'parents' (dict: vertex -> list of parent vertex ids),
             'inputs' (list of input vertex ids, start blue),
             'outputs' (list of output vertex ids, must finish blue)
        fast_memory_budget: int, max number of red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation is in {'load','store','compute','delete'}
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Compute topological order (should already be given, but verify)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo order doesn't match, use our computed one
    if len(topo_order) != n:
        # Fallback: use vertices as given
        topo_order = list(vertices)
    
    # Compute for each vertex: which descendants need it (transitive)
    # Use memoization for efficiency
    needed_by = {}
    
    def compute_needed(v, memo={}):
        if v in memo:
            return memo[v]
        needed = set()
        for c in children[v]:
            needed.add(c)
            needed |= compute_needed(c, memo)
        memo[v] = needed
        return needed
    
    for v in vertices:
        compute_needed(v)
    
    # Count how many times each vertex's value is needed (number of children that need it directly or indirectly)
    # Actually, we need: for each vertex, how many times its value must be available
    # A vertex's value is needed once for each child that uses it
    usage_count = {v: len(children[v]) for v in vertices}
    
    # Strategy: Greedy with lookahead
    # We process vertices in topological order
    # For each vertex, we decide whether to keep it in fast memory
    # We prioritize keeping values that will be needed soon
    
    # First, compute a naive schedule for comparison
    def naive_schedule():
        ops = []
        # Load all inputs
        for v in inputs:
            ops.append(['load', v])
        # Process in topological order
        for v in topo_order:
            if v in inputs:
                continue
            # Check if all parents are in fast memory (we'll track this)
            # For naive: just compute and keep all
            ops.append(['compute', v])
        # Store all outputs
        for v in outputs:
            ops.append(['store', v])
        return ops
    
    naive_ops = naive_schedule()
    naive_transfers = sum(1 for op in naive_ops if op[0] in ('load', 'store'))
    
    # Now implement a better schedule
    # Use a priority queue based on "urgency" - how soon the value will be needed again
    
    # For each vertex, compute the earliest time (in topo order index) when it's needed by a child
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    # For each vertex, compute when its value is last needed
    last_needed = {}
    for v in vertices:
        if not children[v]:
            last_needed[v] = topo_index[v] if v in outputs else -1
        else:
            # The value is needed when the first child is computed
            # But if a child is computed later, we might need to keep it longer
            # For now, use the minimum topo index of children as "first needed"
            child_indices = [topo_index[c] for c in children[v]]
            last_needed[v] = max(child_indices) if child_indices else (topo_index[v] if v in outputs else -1)
    
    # Greedy scheduling with fast memory management
    # State: set of vertices currently in fast memory (red)
    fast_mem = set()
    ops = []
    
    # We need to ensure inputs are loaded before their children are computed
    # Process in topological order
    
    # For each vertex, we need its parents' values available
    # Strategy:
    # 1. Load inputs as needed
    # 2. Compute vertices in topo order
    # 3. Keep values in fast memory based on priority
    # 4. Delete values when fast memory is full and they're least needed
    
    # Priority function: a vertex should be kept if:
    # - It will be needed again soon (has children not yet computed)
    # - It's an output (must be stored at the end)
    
    def priority(v):
        """Higher priority means keep in fast memory"""
        # Count how many children still need this value
        pending_children = 0
        for c in children[v]:
            if c not in computed_set:
                pending_children += 1
        # Outputs get high priority
        if v in outputs:
            return 1000 + pending_children
        return pending_children
    
    computed_set = set()
    loaded = set()  # vertices that have been loaded from disk
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            # Return best effort
            break
        
        if v in inputs:
            # Input: needs to be loaded if not in fast memory
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
                loaded.add(v)
            # Inputs don't need compute
            computed_set.add(v)
            continue
        
        # Non-input vertex: need all parents' values
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                # Need to load parent
                ops.append(['load', p])
                fast_mem.add(p)
                loaded.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        computed_set.add(v)
        
        # Manage fast memory: if over budget, delete lowest priority
        while len(fast_mem) > fast_memory_budget:
            # Find vertex with lowest priority to delete
            # Don't delete if it's still needed by uncomputed children
            candidates = []
            for x in fast_mem:
                # Can we delete x? Only if no uncomputed child needs it
                can_delete = True
                for c in children[x]:
                    if c not in computed_set:
                        can_delete = False
                        break
                if can_delete:
                    candidates.append((priority(x), x))
            
            if not candidates:
                # All vertices are still needed - this shouldn't happen if budget is sufficient
                # Force delete the one with lowest priority among all
                candidates = [(priority(x), x) for x in fast_mem]
            
            if candidates:
                candidates.sort(key=lambda t: t[0])
                _, to_delete = candidates[0]
                ops.append(['delete', to_delete])
                fast_mem.discard(to_delete)
    
    # Store all outputs
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
        else:
            ops.append(['load', v])
            ops.append(['store', v])
    
    # Count transfers
    transfers = sum(1 for op in ops if op[0] in ('load', 'store'))
    
    # If our schedule is worse than naive, use naive
    if transfers > naive_transfers:
        return naive_ops
    
    return ops
# EVOLVE-BLOCK-END
