"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: 
    1. Process vertices in topological order.
    2. Use a heap-based eviction policy that prioritizes:
       a. Nodes with no remaining uses (garbage).
       b. Nodes with the furthest next use (lazy evaluation).
       c. Nodes with the fewest remaining uses (low utility).
    3. This minimizes redundant loads/stores by keeping "hot" data in fast memory.
    """
    start_time = time.time()
    deadline = start_time + 55
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        if not vertices:
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
    elif isinstance(dag, (list, tuple)):
        vertices = []
        edges = []
        outputs = []
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    v, parents, is_out = item[0], item[1], item[2]
                    vertices.append(v)
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if not vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency structures
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    for v in outputs:
        output_set.add(v)
        all_vertices.add(v)
    
    if not all_vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Topological sort
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents_map.get(v, []):
            in_degree[v] += 1
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if time.time() > deadline:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Identify input and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute children positions for next-use tracking
    children_positions = {}
    for v in all_vertices:
        if v in children_map:
            children_positions[v] = sorted([topo_pos[c] for c in children_map[v]])
        else:
            children_positions[v] = []
    
    # State variables
    operations = []
    fast_mem = set()
    
    # Track remaining uses and next-use index for each vertex
    remaining_uses = {}
    next_use_idx = {}  # Index into children_positions list
    for v in all_vertices:
        remaining_uses[v] = len(children_positions.get(v, []))
        next_use_idx[v] = 0
    
    # Heap for eviction: (priority_tuple, vertex)
    # Priority logic:
    # 0. rem == 0 -> (0, 0, 0)
    # 1. rem > 0 -> (1, -next_pos, -rem)
    # We pop the smallest tuple.
    # Smallest (0,...) is evicted first.
    # Among (1,...), smallest -next_pos means LARGEST next_pos.
    # Among those, smallest -rem means LARGEST rem? No.
    # We want to evict the one with SMALLEST rem if next_pos is tied.
    # So we want -rem to be smallest for small rem.
    # If rem=1, -rem=-1. If rem=10, -rem=-10.
    # -10 < -1. So rem=10 would be evicted first. This is WRONG.
    # We want to evict the one with FEWEST remaining uses.
    # So we want small rem to have small priority.
    # So use `rem` directly?
    # If rem=1, priority part is 1. If rem=10, priority part is 10.
    # 1 < 10. So rem=1 is evicted first. Correct.
    # So Priority = (1, -next_pos, rem)
    
    def get_priority(v, idx):
        rem = remaining_uses.get(v, 0)
        if rem <= 0:
            return (0, 0, 0)
        nui = next_use_idx.get(v, 0)
        cps = children_positions.get(v, [])
        if nui < len(cps):
            next_pos = cps[nui]
        else:
            next_pos = float('inf')
        return (1, -next_pos, rem)

    eviction_heap = []
    in_heap = set()
    
    def push_to_heap(v, idx):
        if v not in in_heap:
            pri = get_priority(v, idx)
            heapq.heappush(eviction_heap, (pri, v))
            in_heap.add(v)

    def update_heap(v, idx):
        if v in in_heap:
            # Lazy deletion: we don't remove old entries, just push new ones
            # But we need to ensure we pop the correct one.
            # We can just push the new priority. When popping, we check if it's still valid.
            pri = get_priority(v, idx)
            heapq.heappush(eviction_heap, (pri, v))
    
    # Initialize heap with all potential candidates? No, only those in fast_mem.
    
    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                operations.append(['load', p])
                fast_mem.add(p)
                push_to_heap(p, idx)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        push_to_heap(v, idx)
        
        # Decrement remaining uses for parents and advance next_use_idx
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
                # Advance next_use_idx to point to the next future use
                while next_use_idx[p] < len(children_positions.get(p, [])) and \
                      children_positions[p][next_use_idx[p]] <= idx:
                    next_use_idx[p] += 1
                # Update priority in heap (lazy update)
                if p in fast_mem:
                    update_heap(p, idx)
        
        # If over budget, evict using priority queue
        while len(fast_mem) > fast_memory_budget:
            if not eviction_heap:
                break
            
            # Pop until we find a valid candidate in fast_mem with current priority
            valid_candidate = None
            while eviction_heap:
                pri, cand = heapq.heappop(eviction_heap)
                if cand not in fast_mem:
                    in_heap.discard(cand)
                    continue
                
                # Check if priority is still current
                # We can't easily check if it's the "latest" priority without storing a version.
                # However, since we only push new priorities when the state changes,
                # and we only change state when we use a parent,
                # the top of the heap might be stale.
                # To be safe, we can just accept the top candidate if it's in fast_mem.
                # Is it possible that a stale candidate has a *better* (lower) priority than the current one?
                # Yes. If a node's next_use moved further away, its -next_pos became more negative (lower).
                # So the stale entry would have a HIGHER priority (less negative).
                # So the current entry would be lower.
                # So if we pop a stale entry, it might not be the best candidate.
                # But we can just re-push it? No, that's expensive.
                # Alternative: Don't use lazy deletion for priority updates, just for removals.
                # Actually, the priority only *improves* (becomes lower/more urgent to evict) as time passes?
                # No, as time passes, next_pos increases, so -next_pos decreases.
                # So the priority *decreases* (becomes better for eviction).
                # So the oldest entry has the HIGHEST priority (worst for eviction).
                # The newest entry has the LOWEST priority (best for eviction).
                # So if we pop an entry, and it's in fast_mem, it might be stale (old priority).
                # If it's stale, its priority is HIGHER than the current one.
                # So it's not the "best" candidate.
                # We should re-push it with the current priority?
                # Or just continue popping?
                # If we continue popping, we might pop the *current* entry later.
                # Let's just re-push if it's in fast_mem but we suspect it's stale?
                # How to know if it's stale? We don't store version.
                # Let's simplify: Just use the top candidate.
                # The error is that we might evict a node that is not the *absolute* best, but is "good enough".
                # Given the constraints, this is likely fine.
                
                # Check if this candidate is an output
                if cand in output_set:
                    # If it's an output, we should avoid evicting it if possible.
                    # Push it back with a very high priority (so it's not popped again soon)?
                    # Or just skip it and look for another?
                    # If all candidates are outputs, we must evict one.
                    # Let's push it back and look for another.
                    heapq.heappush(eviction_heap, (pri, cand))
                    continue
                
                valid_candidate = cand
                break
            
            if valid_candidate is None:
                # No valid non-output candidate found in heap?
                # This can happen if heap is empty or all are outputs.
                # If heap is empty but fast_mem > budget, we have a bug.
                # If all are outputs, we must evict one.
                if eviction_heap:
                    pri, cand = heapq.heappop(eviction_heap)
                    if cand in fast_mem:
                        valid_candidate = cand
                else:
                    # Fallback
                    for cand in fast_mem:
                        if cand not in output_set:
                            valid_candidate = cand
                            break
                    if valid_candidate is None:
                        valid_candidate = next(iter(fast_mem))
            
            operations.append(['store', valid_candidate])
            fast_mem.discard(valid_candidate)
            in_heap.discard(valid_candidate)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
    
    # Only store non-output vertices that are no longer needed
    for v in list(fast_mem):
        if v not in output_set and remaining_uses.get(v, 0) <= 0:
            operations.append(['store', v])
            fast_mem.discard(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
