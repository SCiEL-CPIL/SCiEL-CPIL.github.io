import time
import random
import sys

_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph(q):
    # ... standard implementation ...
    pass

def _get_conflict_masks(adj, n):
    # ... standard implementation ...
    pass

def _find_mis(conflict_masks, n):
    # Branch and bound for MIS in conflict graph
    # Returns the mask of the MIS
    pass

def _has_indep_set(adj_masks, mask, k):
    # Branch and bound for independent set of size k in subgraph induced by mask
    pass

def construct():
    # ...
    pass
