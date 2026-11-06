# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _compute_naive_transfers(dag, fast_memory_budget):
    """Compute the naive schedule's transfer count for comparison."""
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [v for v in vertices if v not in parents_map])
    
    transfers = 0
    for v in vertices:
        for p in parents_map.get(v, []):
            transfers += 1  # load parent
        transfers += 1  # store result
    return transfers

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """Greedy pebbling with memory management and recomputation."""
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute recompute cost proxy: number of parents
    parent_count = {v: len(parents_map.get(v, [])) for v in vertices}
    
    # Build children map for usage analysis
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Usage counts: how many times each vertex is needed as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    # Track next use position in topological order for better eviction decisions
    order = _topo_sort(vertices, parents_map)
    order_index = {v: i for i, v in enumerate(order)}
    
    # For each vertex, compute the positions where it's needed as a parent
    next_use_pos = {}
    for v in vertices:
        if children_map[v]:
            child_positions = [order_index[c] for c in children_map[v]]
            next_use_pos[v] = min(child_positions)
        else:
            next_use_pos[v] = float('inf')
    
    fast_memory = set()
    operations = []
    
    def get_eviction_score(v):
        """Lower score = better candidate for eviction.
        
        We want to evict vertices that:
        1. Will be needed soonest (or never) - actually we want to keep those needed soon
        2. Are cheap to reload/recompute
        3. Have low remaining usage
        """
        # If no remaining uses, evict immediately
        r = remaining_usage.get(v, 0)
        if r == 0:
            return (0, 0, 0, 0, v)  # Highest priority to evict
        
        # Distance to next use: larger = less urgent to keep
        dist_to_next = next_use_pos.get(v, float('inf')) - order_index.get(v, 0)
        
        # Type: Input (0) is cheaper to evict than Computed (1)
        # Evicting input: 1 load later = 1 transfer
        # Evicting computed: 1 store now + 1 load later = 2 transfers
        is_comp = 1 if v not in input_set else 0
        
        # Parent count: lower = cheaper to recompute
        pc = parent_count[v]
        
        # Remaining usage: lower = less valuable
        # But we also care about urgency: if used soon, keep it
        
        # Combined score: prioritize evicting vertices that are:
        # - Far from next use (or never used again)
        # - Inputs (cheaper to reload)
        # - Low remaining usage
        # - Low parent count (cheaper to recompute if we delete)
        
        # Normalize distance: cap at some reasonable value
        dist_capped = min(dist_to_next, 100)
        
        return (r, -dist_capped, is_comp, pc, v)
    
    def evict_one():
        """Evict the least valuable vertex from fast memory."""
        if not fast_memory:
            return
        # Find victim with minimum eviction score (most evictable)
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory, evicting if needed."""
        if v in fast_memory:
            return
        
        # Make space if needed
        while len(fast_memory) >= fast_memory_budget:
            if not fast_memory:
                break
            evict_one()
        
        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            # Need to compute v
            # Ensure parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
            
    # Process in topological order
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # If v is not an output and has no remaining usage, evict it immediately
        # to free up space for future vertices
        if v not in output_set and remaining_usage.get(v, 0) == 0:
             # Evict v
             if v in input_set:
                 operations.append(['delete', v])
             else:
                 operations.append(['store', v])
             fast_memory.discard(v)
             
        # Update usage counts for parents of v
        for p in parents_map.get(v, []):
            remaining_usage[p] = remaining_usage.get(p, 0) - 1
            # Update next_use_pos for parent: find next child after current
            child_positions = [order_index[c] for c in children_map[p] if order_index[c] > order_index[v]]
            if child_positions:
                next_use_pos[p] = min(child_positions)
            else:
                next_use_pos[p] = float('inf')

    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
            
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _verify_operations(operations, dag, fast_memory_budget):
    """Verify that a sequence of operations is valid."""
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    
    fast_memory = set()
    
    for op, v in operations:
        if op == 'load':
            if v not in input_set:
                return False  # Can only load inputs
            if v in fast_memory:
                return False  # Already in memory
            if len(fast_memory) >= fast_memory_budget:
                return False  # Budget exceeded
            fast_memory.add(v)
        elif op == 'store':
            if v in input_set:
                return False  # Cannot store inputs
            if v not in fast_memory:
                return False  # Not in memory to store
            fast_memory.discard(v)
        elif op == 'delete':
            if v not in fast_memory:
                return False  # Not in memory to delete
            fast_memory.discard(v)
        elif op == 'compute':
            if v in input_set:
                return False  # Cannot compute inputs
            if v in fast_memory:
                return False  # Already computed and in memory
            # Check all parents are in fast memory
            for p in parents_map.get(v, []):
                if p not in fast_memory:
                    return False
            if len(fast_memory) >= fast_memory_budget:
                return False  # Budget exceeded
            fast_memory.add(v)
        else:
            return False  # Unknown operation
    
    # All outputs must be in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            return False
    
    return True

def _improve_schedule(operations, dag, fast_memory_budget, deadline):
    """Try to improve the schedule by local search."""
    if time.time() > deadline:
        return operations
    
    best_ops = operations
    best_transfers = _count_transfers(operations)
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    
    # Try swapping adjacent operations to see if we can reduce transfers
    # This is a simple local search; we limit iterations to stay within deadline
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(best_ops) - 1):
            # Only consider swapping if both are transfers (load/store/delete)
            # and swapping might help
            if best_ops[i][0] in ('load', 'store', 'delete') and best_ops[i+1][0] in ('load', 'store', 'delete'):
                new_ops = best_ops[:i] + [best_ops[i+1], best_ops[i]] + best_ops[i+2:]
                if _verify_operations(new_ops, dag, fast_memory_budget):
                    new_transfers = _count_transfers(new_ops)
                    if new_transfers < best_transfers:
                        best_ops = new_ops
                        best_transfers = new_transfers
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try the greedy approach
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                # Try to improve with local search
                improved = _improve_schedule(result, dag, fast_memory_budget, deadline)
                if _verify_operations(improved, dag, fast_memory_budget) and len(improved) <= 20000:
                    return improved
                return result
            else:
                # Invalid, fall back to baseline
                from autoresearch_suite.science_tcs_expansion import pebble_baseline
                return pebble_baseline(dag, fast_memory_budget)
        else:
            # Fall back to baseline
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        # Fall back to baseline on any error
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
