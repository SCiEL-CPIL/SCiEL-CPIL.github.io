"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s limit
    
    # Parse DAG structure
    # dag likely has: parents (dict vertex -> list of parent vertices), outputs (list), inputs (list)
    # Need to handle different possible formats
    
    def _extract_structure(dag):
        """Extract DAG structure from various possible formats."""
        if isinstance(dag, dict):
            parents = dag.get('parents', {})
            outputs = dag.get('outputs', [])
            inputs = dag.get('inputs', [])
            # Also try other key names
            if not parents:
                parents = dag.get('preds', dag.get('ancestors', {}))
            if not outputs:
                outputs = dag.get('sinks', dag.get('targets', []))
            if not inputs:
                inputs = dag.get('sources', dag.get('roots', []))
        elif hasattr(dag, 'parents'):
            parents = dag.parents
            outputs = getattr(dag, 'outputs', [])
            inputs = getattr(dag, 'inputs', [])
        else:
            # Try to infer from common structures
            parents = {}
            outputs = []
            inputs = []
            if hasattr(dag, '__dict__'):
                for k, v in dag.__dict__.items():
                    if 'parent' in k.lower() or 'pred' in k.lower():
                        parents = v
                    elif 'output' in k.lower() or 'sink' in k.lower():
                        outputs = v
                    elif 'input' in k.lower() or 'source' in k.lower() or 'root' in k.lower():
                        inputs = v
            # Fallback: try common attribute patterns
            if not parents and hasattr(dag, 'dag'):
                inner = dag.dag
                if isinstance(inner, dict):
                    parents = inner.get('parents', {})
                    outputs = inner.get('outputs', [])
                    inputs = inner.get('inputs', [])
        return parents, outputs, inputs
    
    parents, outputs, inputs = _extract_structure(dag)
    
    # Build full vertex set
    all_vertices = set(inputs) | set(outputs)
    for v, ps in parents.items():
        all_vertices.add(v)
        all_vertices.update(ps)
    
    # Determine which vertices are inputs (no parents or explicitly marked)
    if not inputs:
        inputs = [v for v in all_vertices if v not in parents or not parents.get(v)]
    
    # Determine outputs if not specified
    if not outputs:
        # Outputs are vertices that no other vertex depends on
        all_parent_sets = set()
        for v, ps in parents.items():
            all_parent_sets.update(ps)
        outputs = [v for v in all_vertices if v not in all_parent_sets]
    
    # Topological order
    def topo_sort():
        in_degree = defaultdict(int)
        children = defaultdict(list)
        for v in all_vertices:
            in_degree[v] = 0
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
                in_degree[v] += 1
        
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for w in children[u]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        return order
    
    topo_order = topo_sort()
    
    # Count usage frequency of each vertex
    usage_count = defaultdict(int)
    for v, ps in parents.items():
        for p in ps:
            usage_count[p] += 1
    
    # Strategy: Layer-based pebbling with memory management
    # Group vertices by topological level
    level = {v: 0 for v in inputs}
    for v in topo_order:
        for child in parents:
            if v in parents.get(child, []):
                level[child] = max(level.get(child, 0), level[v] + 1)
    
    # Build levels
    levels = defaultdict(list)
    for v in topo_order:
        levels[level[v]].append(v)
    
    max_level = max(levels.keys()) if levels else 0
    
    # Fast memory simulation
    fast_mem = set()  # vertices currently in fast memory (red or blue)
    fast_mem_blue = set()  # vertices in fast memory that are blue (stored)
    operations = []
    
    # Inputs start blue - they're in slow memory, need to be loaded
    # Strategy: Load inputs as needed, keep in fast memory if budget allows
    
    # Simple greedy: process in topological order
    # For each vertex, ensure parents are available (load if not in fast mem)
    # Compute, then decide whether to keep in fast mem or store
    
    # Better strategy: Use the fast memory budget to keep high-usage vertices
    # Sort by usage frequency for memory allocation
    
    # Initialize: all inputs are blue in slow memory
    # We'll track what's in fast memory
    
    # Greedy approach:
    # 1. Load input when first needed
    # 2. Keep in fast memory if budget allows and usage > 1
    # 3. Compute vertex
    # 4. Store output at end
    # 5. Delete intermediate when no longer needed
    
    # Track when each vertex is last used
    last_use = {}
    for i, v in enumerate(topo_order):
        last_use[v] = i
        for child in parents:
            if v in parents.get(child, []):
                last_use[v] = max(last_use[v], last_use.get(child, i))
    
    # Actually, let me use a simpler but effective approach:
    # Process in topological order
    # Load each parent when needed (if not already in fast memory)
    # Compute
    # Keep result in fast memory if it will be used again and budget allows
    # Otherwise store to slow memory
    # Delete from fast memory when no longer needed and budget is tight
    
    # Track state
    in_fast = set()  # vertices currently in fast memory
    in_slow_blue = set(inputs)  # vertices available in slow memory as blue
    
    # Pre-compute: for each vertex, list of children that need it
    children_of = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children_of[p].append(v)
    
    # Count remaining uses
    remaining_uses = dict(usage_count)
    for v in inputs:
        remaining_uses[v] = usage_count[v]
    
    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        # If v is an input, it might already be loaded
        # Ensure all parents are available
        ps = parents.get(v, [])
        for p in ps:
            if p not in in_fast:
                # Need to load p
                if p in in_slow_blue:
                    operations.append(['load', p])
                    in_fast.add(p)
                else:
                    # p should have been computed and either in fast or slow
                    # If not in fast, load from slow
                    operations.append(['load', p])
                    in_fast.add(p)
        
        # Compute v
        operations.append(['compute', v])
        
        # Decide whether to keep v in fast memory
        # v is now in fast memory (just computed)
        in_fast.add(v)
        
        # Update remaining uses for parents
        for p in ps:
            remaining_uses[p] = max(0, remaining_uses[p] - 1)
            if remaining_uses[p] == 0 and p not in in_fast:
                pass  # already not in fast
            elif remaining_uses[p] == 0:
                # No longer needed, can delete if budget tight
                pass
        
        # Check if we're over budget
        if len(in_fast) > fast_memory_budget:
            # Need to delete some vertices
            # Delete vertices that are no longer needed first
            to_delete = []
            for u in in_fast:
                if remaining_uses.get(u, 0) == 0 and u not in outputs:
                    to_delete.append(u)
            
            # Sort by remaining uses (delete lowest first)
            to_delete.sort(key=lambda u: remaining_uses.get(u, 0))
            
            for u in to_delete:
                if len(in_fast) <= fast_memory_budget:
                    break
                operations.append(['delete', u])
                in_fast.remove(u)
                in_slow_blue.add(u)  # stored in slow memory
        
        # If v is an output, ensure it ends up blue in slow memory
        # (it will be stored at the end)
    
    # Store all outputs
    for o in outputs:
        if o in in_fast:
            operations.append(['store', o])
            in_fast.remove(o)
            in_slow_blue.add(o)
        elif o not in in_slow_blue:
            # Shouldn't happen, but handle gracefully
            pass
    
    # Truncate if over limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END

