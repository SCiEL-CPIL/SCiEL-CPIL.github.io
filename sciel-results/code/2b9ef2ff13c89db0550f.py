# EVOLVE-BLOCK-START
import time
import sys
import json

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Args:
        dag: DAG structure 
        fast_memory_budget: number of fast memory slots
    
    Returns:
        list of [operation, vertex] where operation is load/store/compute/delete
    """
    start_time = time.time()
    deadline = start_time + 85
    
    try:
        # Diagnostic: print dag structure to stderr (first call only)
        if not hasattr(pebble, '_diag_done'):
            pebble._diag_done = True
            try:
                diag_info = _describe_dag(dag)
                sys.stderr.write(f"DAG DIAG: {diag_info}\n")
            except Exception as e:
                sys.stderr.write(f"Diag error: {e}\n")
        
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        
        if time.time() > deadline:
            return result
            
        # Try improved algorithm
        improved = _improved_pebble(dag, fast_memory_budget, deadline)
        if improved is not None:
            return improved
        return result
    except Exception as e:
        sys.stderr.write(f"Main error: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []


def _describe_dag(dag):
    """Describe the DAG structure for diagnostics."""
    if isinstance(dag, dict):
        keys = list(dag.keys())
        desc = f"dict with keys: {keys}"
        # Try to get sizes
        for k in keys:
            v = dag[k]
            if isinstance(v, (list, tuple)):
                desc += f"\n  {k}: list/tuple of len {len(v)}"
                if len(v) > 0:
                    desc += f", first elem: {repr(v[0])[:100]}"
            elif isinstance(v, dict):
                desc += f"\n  {k}: dict with {len(v)} keys"
            else:
                desc += f"\n  {k}: {type(v).__name__} = {repr(v)[:100]}"
        return desc
    elif isinstance(dag, (list, tuple)):
        return f"list/tuple of len {len(dag)}"
    else:
        return f"type={type(dag).__name__}, repr={repr(dag)[:200]}"


def _improved_pebble(dag, fast_memory_budget, deadline):
    """
    Improved pebbling algorithm.
    Returns None if we can't improve, or a list of operations.
    """
    # Placeholder - need to understand dag format first
    return None
# EVOLVE-BLOCK-END
