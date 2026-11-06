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

def _greedy_pebble_heap(dag, fast_memory_budget, deadline):
    """
    Heap-based greedy pebbling.
    Prioritizes processing vertices that are ready (all parents resolved) 
    and have high 'value' (high remaining usage, expensive to recompute).
    """
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
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Pre-compute metrics
    # remaining_usage: how many children still need this vertex
    remaining_usage = {v: len(children_map[v]) for v in vertices}
    
    # in_degree: number of unprocessed parents
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    
    # depth: for tie-breaking (prefer deeper? or shallower?)
    # Let's use depth to break ties.
    depth = {}
    for v in vertices:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            # Approximate depth using topological order index
            pass
            
    order = _topo_sort(vertices, parents_map)
    order_index = {v: i for i, v in enumerate(order)}
    
    # Fast memory state
    fast_memory = set()
    # State: 'in_fast', 'in_slow', 'gone'
    # Inputs start in 'in_slow' (available to load)
    # Computed start in 'gone' (must compute)
    state = {v: 'in_slow' if v in input_set else 'gone' for v in vertices}
    
    operations = []
    
    # Heap items: (-priority, -order_index, vertex)
    # We want to process high priority first.
    # Priority: remaining_usage * 10 + (0 if input else 1) ?
    # Actually, we want to process vertices that *unlock* many others or are high value.
    # Let's prioritize vertices with high remaining_usage of their *parents*?
    # No, just process ready vertices.
    
    ready_heap = []
    
    # Initialize ready vertices (inputs)
    for v in input_vertices:
        # Priority: high remaining_usage is good? 
        # If we process an input, we load it.
        # If it has high fan-out, we should probably keep it.
        # But processing it doesn't change its state in fast memory, just makes children ready.
        # Wait, "compute" is the operation for non-inputs.
        # For inputs, we just "load" them when needed?
        # The task says: "Inputs start blue... every red value consumes one fast-memory slot".
        # Usually, you load an input when you need to compute a child.
        # So "ready" vertices are those whose parents are *available* (in fast or slow).
        # But to compute, parents must be in *fast*.
        
        # So, a vertex is "Ready to Compute" if all its parents are in Fast Memory.
        # This is different from "Ready to Load".
        
        # Let's change strategy:
        # 1. Identify "Active" vertices: those in Fast Memory.
        # 2. Identify "Candidate" vertices: those whose parents are all in Fast Memory (or are inputs that we can load).
        
        # This is getting complex. Let's stick to Topological Order but with better eviction.
        pass

    # Revised Plan: Stick to Topological Order but improve Eviction.
    # The previous attempt was 1.0.
    # Let's try to **minimize transfers** by keeping a **working set** of high-usage parents.
    
    # We will process in Topological Order.
    # We maintain a set of "Protected" vertices: those with high remaining_usage.
    # We never evict Protected vertices if possible.
    
    protected = set()
    # Identify vertices with high fan-out (top 10%?)
    # Or simply: keep vertices if remaining_usage > 1.
    
    # Let's use the Heap for *Eviction* decisions, but process in Topo Order.
    
    # Re-define ensure_in_memory to be smarter.
    
    fast_mem = set()
    ops = []
    
    def get_eviction_priority(v):
        """
        Returns a tuple for eviction. Lower is evicted first.
        We want to evict:
        1. Vertices with 0 remaining usage (must evict).
        2. Vertices with low remaining usage.
        3. Vertices that are cheap to restore (Inputs).
        4. Vertices that are far from next use.
        """
        r = remaining_usage.get(v, 0)
        if r == 0:
            return (0, 0, 0, 0, v)
        
        # Distance to next use
        # We need to know the next child in topological order.
        # Pre-compute next_use_index for each vertex.
        
        # Type: Input (0) vs Computed (1). 
        # Evict Inputs first (cheaper to reload).
        is_comp = 1 if v not in input_set else 0
        
        # Parent count (proxy for recompute cost)
        pc = len(parents_map.get(v, []))
        
        # Remaining usage
        return (r, is_comp, pc, v)

    # Pre-compute next_use_index
    # For each vertex, the index of its first child in topological order.
    next_use_idx = {}
    for v in vertices:
        if children_map[v]:
            # Find min order_index of children
            min_idx = min(order_index[c] for c in children_map[v])
            next_use_idx[v] = min_idx
        else:
            next_use_idx[v] = float('inf')
            
    # We need to update next_use_idx as we process.
    # This is expensive. 
    # Alternative: Just use remaining_usage.
    
    def evict_one():
        if not fast_mem:
            return
        # Find victim
        # We want to evict the one with lowest "Value".
        # Value = remaining_usage * (1 if input else 2) ?
        # Let's just use remaining_usage, then type.
        
        # To avoid O(N) scan every time, we could use a heap, but state changes.
        # Given small N (likely < 1000), O(N) is fine.
        
        victim = min(fast_mem, key=lambda v: (remaining_usage.get(v, 0), 0 if v in input_set else 1, len(parents_map.get(v, [])), v))
        
        if remaining_usage.get(victim, 0) == 0:
            # No more uses, just delete
            ops.append(['delete', victim])
        else:
            # Has more uses
            if victim in input_set:
                ops.append(['delete', victim])
            else:
                ops.append(['store', victim])
        
        fast_mem.discard(victim)
        state[victim] = 'in_slow' if victim in input_set else 'stored'

    def ensure_in_memory(v):
        if v in fast_mem:
            return
        
        # Make space
        while len(fast_mem) >= fast_memory_budget:
            evict_one()
            
        if v in input_set:
            ops.append(['load', v])
            fast_mem.add(v)
            state[v] = 'in_fast'
        else:
            # Check if it was stored
            if state.get(v) == 'stored':
                ops.append(['load', v])
                fast_mem.add(v)
                state[v] = 'in_fast'
            else:
                # Must compute
                # Ensure parents are in memory
                for p in parents_map.get(v, []):
                    ensure_in_memory(p)
                ops.append(['compute', v])
                fast_mem.add(v)
                state[v] = 'in_fast'

    # Process in Topological Order
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # After computing/loading v, it is in fast memory.
        # Update remaining_usage of parents?
        # No, remaining_usage is "how many children still need this vertex".
        # When we process v, we satisfy one need for its parents.
        for p in parents_map.get(v, []):
            remaining_usage[p] -= 1
            
        # If v is not an output and has no remaining usage, evict it?
        # It might be needed as a parent for a later vertex?
        # No, if remaining_usage[v] == 0, no later vertex needs it.
        # But wait, remaining_usage[v] is the number of children.
        # All children of v are later in topological order.
        # So if we are processing v, and we haven't processed its children yet,
        # remaining_usage[v] is the count of unprocessed children.
        
        # So we should NOT evict v if remaining_usage[v] > 0.
        # We only evict v when we need space for something else.
        
        # However, if v is an input, and it has 0 remaining usage (all children processed),
        # we can evict it.
        
    # Ensure outputs are in memory
    for o in outputs:
        if o not in fast_mem:
            ensure_in_memory(o)
            
    return ops

def _count_transfers(operations):
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80
    
    try:
        result = _greedy_pebble_heap(dag, fast_memory_budget, deadline)
        
        if result and len(result) <= 20000:
            # Verify validity loosely
            return result
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
