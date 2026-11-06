"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _parse_dag(dag):
    """Parse the DAG structure into a usable format."""
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', len(dag.get('parents', {}))))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('output_vertices', []))
        return n, parents, outputs
    
    if isinstance(dag, (tuple, list)):
        if len(dag) >= 3:
            return dag[0], dag[1], dag[2]
        elif len(dag) == 2:
            return dag[0], dag[1], []
    
    if isinstance(dag, list):
        n = len(dag)
        parents = {i: (list(dag[i]) if i < n else []) for i in range(n)}
        outputs = [n - 1] if n > 0 else []
        return n, parents, outputs
    
    raise ValueError(f"Unrecognized DAG format: {type(dag)}")

def _get_children(parents, n):
    """Build children list from parents dict."""
    children = [[] for _ in range(n)]
    for v, par_list in parents.items():
        for p in par_list:
            if p < n:
                children[p].append(v)
    return children

def _simulate_pebbling(vertices, parents, n, fast_memory_budget, children, outputs_set, initial_fast_mem):
    """Simulate a pebbling strategy and return operations.
    
    Uses a greedy approach with proper fast memory management:
    - Track what's in fast memory (red pebbles)
    - When computing v, ensure all parents are in fast memory
    - After computing, decide whether to keep v in fast memory or delete it
    - Use LRU-like eviction when fast memory is full
    """
    if not vertices:
        return []
    
    vset = set(vertices)
    ops = []
    
    # State tracking
    fast_mem = set(initial_fast_mem)  # vertices currently in fast memory
    computed = set()  # vertices that have been computed at least once
    in_fast = set(fast_mem)  # same as fast_mem for clarity
    
    # Topological sort
    sub_parents = {v: [p for p in parents.get(v, []) if p in vset] for v in vertices}
    in_degree = {v: len(sub_parents[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            if c in vset:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    # Count how many times each vertex is needed as a parent
    parent_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in sub_parents[v]:
            parent_count[p] += 1
    
    # Track remaining uses
    remaining_uses = parent_count.copy()
    
    # Process in topological order
    for v in topo:
        # Ensure all parents are in fast memory
        for p in sub_parents[v]:
            if p not in fast_mem:
                # Need to load p from slow memory
                ops.append(['load', p])
                fast_mem.add(p)
            
            # Check if fast memory is full
            if len(fast_mem) > fast_memory_budget:
                # Evict a vertex that's no longer needed or least recently used
                # Simple heuristic: evict a vertex with 0 remaining uses
                evicted = None
                for candidate in list(fast_mem):
                    if candidate != v and remaining_uses.get(candidate, 0) == 0:
                        evicted = candidate
                        break
                
                if evicted is None:
                    # Evict the first non-output vertex
                    for candidate in list(fast_mem):
                        if candidate not in outputs_set and candidate != v:
                            evicted = candidate
                            break
                
                if evicted is not None:
                    ops.append(['delete', evicted])
                    fast_mem.remove(evicted)
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        fast_mem.add(v)
        
        # Decrement remaining uses of parents
        for p in sub_parents[v]:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
        
        # Decide whether to keep v in fast memory
        if v in outputs_set:
            # Must keep outputs
            pass
        elif remaining_uses.get(v, 0) == 0:
            # No more uses, delete immediately
            ops.append(['delete', v])
            fast_mem.remove(v)
        else:
            # Keep in fast memory for now
            pass
    
    # At the end, ensure all outputs are in fast memory
    for v in outputs_set:
        if v in vset and v not in fast_mem:
            ops.append(['load', v])
            fast_mem.add(v)
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    try:
        n, parents, outputs = _parse_dag(dag)
        outputs_set = set(outputs)
        children = _get_children(parents, n)
        
        all_vertices = list(range(n))
        
        # Use linear topological pebbling with fast memory management
        ops = _simulate_pebbling(all_vertices, parents, n, fast_memory_budget, children, outputs_set, set())
        
        # Ensure we don't exceed operation limit
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
    
    except Exception:
        # Ultimate fallback: use baseline if available
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            # Last resort: minimal valid schedule
            try:
                n, parents, outputs = _parse_dag(dag)
                children = _get_children(parents, n)
                ops = _simulate_pebbling(list(range(n)), parents, n, fast_memory_budget, children, set(outputs))
                return ops[:20000]
            except Exception:
                return []
# EVOLVE-BLOCK-END
