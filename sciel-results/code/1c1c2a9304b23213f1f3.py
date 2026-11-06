"""Optimized DAG pebbling with LRU eviction and lookahead."""
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling minimizing transfers using greedy topological order with
    LRU + lookahead eviction policy.
    
    Strategy:
    1. Process vertices in topological order
    2. Load parents into fast memory as needed
    3. Compute vertex, store result in fast memory
    4. Evict vertices using combined LRU + next-use-distance heuristic
    5. At the end, ensure all outputs are in fast memory
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    # --- Parse DAG structure ---
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        edges = dag.get('edges', dag.get('arcs', []))
        outputs = dag.get('outputs', dag.get('sinks', []))
        # Check for parent-based format: each vertex is (name, parents, is_output)
        if vertices and isinstance(vertices[0], (list, tuple)):
            parents_list = []
            outputs_list = []
            all_v = []
            for item in vertices:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 3:
                        v, par, is_out = item[0], item[1], item[2]
                        all_v.append(v)
                        if is_out:
                            outputs_list.append(v)
                        for p in par:
                            parents_list.append((v, p))
                    elif len(item) == 2:
                        v, par = item[0], item[1]
                        all_v.append(v)
                        for p in par:
                            parents_list.append((v, p))
            if all_v:
                # Rebuild from parent format
                all_vertices = set(all_v)
                for v, p in parents_list:
                    all_vertices.add(v)
                    all_vertices.add(p)
                topo_order = list(all_v)  # assume given in topological order
                # Verify and add any missing
                in_set = set(topo_order)
                for av in all_vertices:
                    if av not in in_set:
                        topo_order.append(av)
                parents_map = defaultdict(list)
                children_map = defaultdict(list)
                for v, p in parents_list:
                    parents_map[v].append(p)
                    children_map[p].append(v)
                output_set = set(outputs_list)
                # Fallback: outputs = sinks
                if not output_set:
                    output_set = {v for v in all_vertices if not children_map.get(v)}
                n = len(topo_order)
                topo_pos = {v: i for i, v in enumerate(topo_order)}
                return _run_schedule(topo_order, parents_map, children_map,
                                     output_set, fast_memory_budget, deadline)
    
        if not vertices:
            return []
        
        parents_map = defaultdict(list)
        children_map = defaultdict(list)
        all_vertices = set()
        
        for v in vertices:
            all_vertices.add(v)
        
        for edge in edges:
            if len(edge) >= 2:
                p, c = edge[0], edge[1]
                parents_map[c].append(p)
                children_map[p].append(c)
                all_vertices.add(p)
                all_vertices.add(c)
        
        output_set = set(outputs)
        if not output_set:
            output_set = {v for v in all_vertices if not children_map.get(v)}
        
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
        
        if not vertices:
            return []
        
        parents_map = defaultdict(list)
        children_map = defaultdict(list)
        all_vertices = set()
        
        for v in vertices:
            all_vertices.add(v)
        
        for edge in edges:
            if len(edge) >= 2:
                p, c = edge[0], edge[1]
                parents_map[c].append(p)
                children_map[p].append(c)
                all_vertices.add(p)
                all_vertices.add(c)
        
        output_set = set(outputs)
        if not output_set:
            output_set = {v for v in all_vertices if not children_map.get(v)}
    else:
        return []
    
    if not all_vertices:
        return []
    
    # --- Topological sort ---
    in_degree = {v: len(parents_map.get(v, [])) for v in all_vertices}
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
        # Fallback: use insertion order
        topo_order = list(all_vertices)
    
    return _run_schedule(topo_order, parents_map, children_map,
                         output_set, fast_memory_budget, deadline)


def _run_schedule(topo_order, parents_map, children_map, output_set,
                  fast_memory_budget, deadline):
    """Execute the pebbling schedule with optimized eviction."""
    n = len(topo_order)
    if n == 0:
        return []
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute children positions for each vertex
    children_pos = {}
    for v in topo_order:
        ch = children_map.get(v, [])
        if ch:
            children_pos[v] = sorted([topo_pos[c] for c in ch])
        else:
            children_pos[v] = []
    
    # Track next-use pointer
    child_ptr = {v: 0 for v in topo_order}
    
    # Remaining uses
    remaining_uses = {v: len(children_pos.get(v, [])) for v in topo_order}
    
    # State tracking
    state = {}  # 'fast', 'slow', 'uncomputed'
    fast_mem = set()
    last_access = {}  # for LRU
    
    operations = []
    op_count = 0
    
    def get_next_use(v, current_idx):
        """Get next position where v is needed as a parent."""
        cps = children_pos.get(v, [])
        ptr = child_ptr.get(v, 0)
        while ptr < len(cps) and cps[ptr] <= current_idx:
            ptr += 1
        child_ptr[v] = ptr
        if ptr < len(cps):
            return cps[ptr]
        return None
    
    def eviction_score(v, current_idx):
        """
        Score a candidate for eviction. Lower score = evict first.
        Combines:
        - next use distance (closer = keep)
        - remaining uses (more = keep)
        - whether it's an output (outputs get bonus)
        - LRU recency
        """
        rem = remaining_uses.get(v, 0)
        
        if rem <= 0:
            # No more uses - very safe to evict
            base = -1000.0
        else:
            next_pos = get_next_use(v, current_idx)
            if next_pos is not None:
                dist = next_pos - current_idx
                # Exponential decay: closer uses are much more important
                importance = rem / (1.0 + dist * 0.5)
                base = importance
            else:
                base = 0.5 * rem
        
        # Output bonus: harder to evict
        if v in output_set:
            base *= 3.0
        
        # LRU penalty: recently used is harder to evict
        last = last_access.get(v, -1)
        age = current_idx - last
        lru_factor = 1.0 / (1.0 + age * 0.1)
        
        return base * lru_factor
    
    # --- Main loop ---
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        if op_count >= 19990:  # Leave room for finalization
            break
        
        # Load parents into fast memory
        for p in parents_map.get(v, []):
            if state.get(p) != 'fast':
                operations.append(['load', p])
                op_count += 1
                state[p] = 'fast'
                fast_mem.add(p)
                last_access[p] = idx
            
            # Update access
            last_access[p] = idx
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        state[v] = 'fast'
        fast_mem.add(v)
        last_access[v] = idx
        
        # Decrement remaining uses of parents
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            if op_count >= 19990:
                break
            
            # Find best eviction candidate
            best_v = None
            best_score = float('inf')
            
            for cand in list(fast_mem):
                # Never evict if it's still needed right now (parent of current or future close)
                if remaining_uses.get(cand, 0) == 0:
                    sc = -9999.0
                else:
                    sc = eviction_score(cand, idx)
                
                if sc < best_score:
                    best_score = sc
                    best_v = cand
            
            if best_v is None:
                break
            
            operations.append(['store', best_v])
            op_count += 1
            state[best_v] = 'slow'
            fast_mem.discard(best_v)
    
    # --- Finalization: ensure all outputs are in fast memory ---
    for v in topo_order:
        if time.time() > deadline or op_count >= 20000:
            break
        if v in output_set and state.get(v) != 'fast':
            operations.append(['load', v])
            op_count += 1
            state[v] = 'fast'
            fast_mem.add(v)
            last_access[v] = n
    
    # Store non-output vertices from fast memory (optional cleanup)
    for v in list(fast_mem):
        if time.time() > deadline or op_count >= 20000:
            break
        if v not in output_set and state.get(v) == 'fast':
            operations.append(['store', v])
            op_count += 1
            state[v] = 'slow'
            fast_mem.discard(v)
    
    return operations
