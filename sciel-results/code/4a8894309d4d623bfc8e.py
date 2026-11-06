<<<<<<< SEARCH
def _pebble_no_evict(dag, budget):
    """Schedule when budget >= n: load inputs, compute in topo order, store outputs."""
    n = dag["n_vertices"]
    parents = dag["parents"]
    outputs = dag["output_vertices"]
    output_set = set(outputs)
=======
def _pebble_no_evict(dag, budget):
    """Schedule when budget >= n: load inputs, compute in topo order, store outputs."""
    parents = dag["parents"]
    n = len(parents)
    outputs = dag["output_vertices"]
    output_set = set(outputs)
>>>>>>> REPLACE
