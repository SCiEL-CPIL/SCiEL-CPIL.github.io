"""Program for the Witt design S(5,6,12) packing task.

Returns a list of distinct 6-subsets of 0..11 such that no 5-subset lies in
two blocks, maximizing the number of blocks. The optimum 132 is attained
exactly by the Steiner system S(5,6,12).
"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Hard wall-clock limit from the evaluator is 150 s; keep a safety margin.
_DEADLINE_START = time.time()
_DEADLINE_SECONDS = 140.0


def _now():
    return time.time()


def _time_left():
    return _DEADLINE_SECONDS - (_now() - _DEADLINE_START)


def construct():
    """Return a list of distinct 6-subsets of 0..11 (as lists) that form a
    packing with no repeated 5-subset, maximizing the number of blocks.

    Uses a deterministic lexicographic greedy algorithm which is known to
    attain the Witt design S(5,6,12) with 132 blocks.
    """
    covered = set()
    blocks = []
    
    # Iterate through all 6-subsets in lexicographic order
    for b in combinations(range(12), 6):
        # Check if any 5-subset of b is already covered
        if all(s not in covered for s in combinations(b, 5)):
            blocks.append(list(b))
            covered.update(combinations(b, 5))
            
    return blocks
# EVOLVE-BLOCK-END
