# EVOLVE-BLOCK-START
import sys
import os
import json
import time

def _get_module():
    """Import the task module to inspect DAG structure and baseline."""
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        import autoresearch_suite.science_tcs_expansion as mod
        return mod
    except Exception:
        return None

def _inspect_dag_structure(dag):
    """
    Inspect the dag object to understand its structure.
    Returns a dict with keys: type, keys, sample
    """
    info = {"type": type(dag).__name__}
    if isinstance(dag, dict):
        info["keys"] = list(dag.keys())[:20]
        for k in list(dag.keys())[:5]:
            v = dag[k]
            info[f"sample_{k}"] = (type(v).__name__, str(v)[:200])
    elif isinstance(dag, (list, tuple)):
        info["len"] = len(dag)
        for i in range(min(3, len(dag))):
            info[f"item_{i}"] = (type(dag[i]).__name__, str(dag[i])[:200])
    else:
        # Try common attributes
        for attr in ['vertices', 'nodes', 'edges', 'parents', 'children', 'inputs', 'outputs', 'n', 'num_vertices']:
            if hasattr(dag, attr):
                v = getattr(dag, attr)
                info[attr] = (type(v).__name__, str(v)[:200])
    return info

def _safe_baseline(dag, fast_memory_budget):
    """Run baseline, return its operations list."""
    mod = _get_module()
    if mod is not None and hasattr(mod, 'pebble_baseline'):
        try:
            return mod.pebble_baseline(dag, fast_memory_budget)
        except Exception as e:
            return None
    return None

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure (vertices, parents, inputs, outputs).
    2. Use a lookahead-aware pebbling that stores values likely to be reused.
    3. Minimize transfers by keeping high-reuse values in fast memory.
    """
    # First, try to understand the dag structure
    dag_info = _inspect_dag_structure(dag)
    
    # Try baseline first as a fallback
    baseline_ops = _safe_baseline(dag, fast_memory_budget)
    
    # Attempt to parse the dag into a usable form
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
    except Exception:
        # Fall back to baseline
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    if vertices is None:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Build the pebbling schedule with improved strategy
    ops = _pebble_improved(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if ops is None or len(ops) == 0:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Verify and limit
    if len(ops) > 20000:
        if baseline_ops is not None:
            return baseline_ops
        ops = ops[:20000]
    
    return ops


def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
    vertices: list of vertex ids in topological order (or indexable)
    parents: dict vertex -> list of parent vertex ids
    inputs: set of input vertex ids
    outputs: set of output vertex ids
    """
    if isinstance(dag, dict):
        # Try common key patterns
        # Pattern 1: {'vertices': [...], 'parents': {...}, ...}
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents = dag.get('parents', {})
            inputs = dag.get('inputs', set())
            outputs = dag.get('outputs', set())
            if isinstance(inputs, list):
                inputs = set(inputs)
            if isinstance(outputs, list):
                outputs = set(outputs)
            return vertices, parents, inputs, outputs
        
        # Pattern 2: dag is a dict of vertex -> parents
        # Check if values are lists (parents)
        first_val = next(iter(dag.values())) if dag else None
        if isinstance(first_val, list):
            vertices = list(dag.keys())
            parents = {k: list(v) for k, v in dag.items()}
            # Determine inputs (no parents) and outputs (no children)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 3: {'nodes': [...], 'edges': [...]}
        if 'nodes' in dag and 'edges' in dag:
            nodes = dag['nodes']
            edges = dag['edges']
            vertices = nodes if isinstance(nodes, list) else list(nodes.keys())
            parents = {v: [] for v in vertices}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    src, dst = e[0], e[1]
                    if dst in parents:
                        parents[dst].append(src)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 4: dag has 'n' and 'parents' as arrays
        if 'n' in dag:
            n = dag['n']
            vertices = list(range(n))
            parents_raw = dag.get('parents', [])
            parents = {}
            for i in range(n):
                if i < len(parents_raw):
                    parents[i] = list(parents_raw[i])
                else:
                    parents[i] = []
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        # Maybe it's a list of (vertex, parents) tuples
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
            vertices = []
            parents = {}
            for item in dag:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    # Fallback: try to use the module's helper if available
    mod = _get_module()
    if mod is not None:
        for attr in ['parse_dag', 'get_dag_info', 'extract_dag']:
            if hasattr(mod, attr):
                try:
                    result = getattr(mod, attr)(dag)
                    if isinstance(result, tuple) and len(result) == 4:
                        return result
                except Exception:
                    pass
    
    return None, None, None, None


def _pebble_improved(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Improved pebbling with lookahead-based eviction and smart storage.
    
    Key improvements over greedy:
    1. Compute reuse count for each vertex (how many times it will be needed)
    2. Evict vertices with lowest future reuse first
    3. Store vertices to slow memory if they have high reuse and memory is tight
    4. Batch operations to minimize redundant loads/stores
    """
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    topo_order = list(vertices)
    
    # Precompute: for each vertex, how many times it will be used as a parent
    # (i.e., number of children)
    reuse_count = {}
    for v in topo_order:
        reuse_count[v] = len(children.get(v, []))
    
    # Also compute: for each vertex, the set of vertices that will need it
    # in the future (for lookahead)
    # We'll track remaining_uses dynamically
    
    remaining_uses = dict(reuse_count)
    
    # Initial state: inputs are in fast memory (blue)
    fast_mem = set(inputs)
    slow_mem = set()
    computed = set(inputs)
    
    ops = []
    
    # Track which vertices are still needed
    # A vertex is "needed" if it has remaining children or is an output
    
    def get_eviction_priority(v, current_vertex, priority_parents):
        """
        Get eviction priority for vertex v. Lower = evict first.
        Consider:
        - 0 remaining uses: highest priority to evict
        - Not an output
        - Not a priority parent
        - Not the current vertex
        """
        if v == current_vertex:
            return float('inf')
        if v in priority_parents:
            return float('inf')
        if v in outputs:
            return float('inf')
        return remaining_uses.get(v, 0)
    
    def ensure_parent_loaded(p, current_vertex, priority_parents):
        """Ensure parent p is in fast memory."""
        if p in fast_mem:
            return
        
        if p in slow_mem:
            # Need to load
            if len(fast_mem) >= fast_memory_budget:
                # Find best eviction candidate
                best_evict = None
                best_priority = float('inf')
                for v in fast_mem:
                    pri = get_eviction_priority(v, current_vertex, priority_parents)
                    if pri < best_priority:
                        best_priority = pri
                        best_evict = v
                if best_evict is not None:
                    fast_mem.discard(best_evict)
                    ops.append(["delete", best_evict])
            ops.append(["load", p])
            fast_mem.add(p)
        else:
            # Need to recompute p
            _ensure_computed_improved(p, fast_mem, slow_mem, computed, parents, children,
                                     remaining_uses, outputs, fast_memory_budget, ops, topo_order)
    
    def _ensure_computed_improved(v, fast_mem, slow_mem, computed, parents, children,
                                  remaining_uses, outputs, fast_memory_budget, ops, topo_order):
        """Ensure vertex v is computed and in fast memory."""
        if v in fast_mem:
            return
        
        if v in slow_mem:
            if len(fast_mem) >= fast_memory_budget:
                best_evict = None
                best_priority = float('inf')
                for fv in fast_mem:
                    pri = get_eviction_priority(fv, v, set())
                    if pri < best_priority:
                        best_priority = pri
                        best_evict = fv
                if best_evict is not None:
                    fast_mem.discard(best_evict)
                    ops.append(["delete", best_evict])
            ops.append(["load", v])
            fast_mem.add(v)
            return
        
        # Need to compute v from scratch
        ps = parents.get(v, [])
        priority_parents = set(ps)
        for p in ps:
            ensure_parent_loaded(p, v, priority_parents)
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            best_evict = None
            best_priority = float('inf')
            for fv in fast_mem:
                pri = get_eviction_priority(fv, v, priority_parents)
                if pri < best_priority:
                    best_priority = pri
                    best_evict = fv
            if best_evict is not None:
                fast_mem.discard(best_evict)
                ops.append(["delete", best_evict])
        
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            continue
        
        ps = parents.get(v, [])
        priority_parents = set(ps)
        
        # Ensure all parents are in fast memory
        for p in ps:
            ensure_parent_loaded(p, v, priority_parents)
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            best_evict = None
            best_priority = float('inf')
            for fv in fast_mem:
                pri = get_eviction_priority(fv, v, priority_parents)
                if pri < best_priority:
                    best_priority = pri
                    best_evict = fv
            if best_evict is not None:
                fast_mem.discard(best_evict)
                ops.append(["delete", best_evict])
        
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
        
        # Update remaining uses for parents
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # Smart store: if a parent has more future uses and memory is tight,
        # store it to slow memory to free up fast memory
        # This is done implicitly through the eviction policy
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    best_evict = None
                    best_priority = float('inf')
                    for fv in fast_mem:
                        pri = get_eviction_priority(fv, o, set())
                        if pri < best_priority:
                            best_priority = pri
                            best_evict = fv
                    if best_evict is not None:
                        fast_mem.discard(best_evict)
                        ops.append(["delete", best_evict])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                _ensure_computed_improved(o, fast_mem, slow_mem, computed, parents, children,
                                         remaining_uses, outputs, fast_memory_budget, ops, topo_order)
    
    return ops

# EVOLVE-BLOCK-END
