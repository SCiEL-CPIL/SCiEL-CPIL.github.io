import time
from itertools import combinations

# EVOLVE-BLOCK-START
MOD = 101
DENOMS = list(range(1, 11))

# Precompute modular inverses
_inv = [0] * 11
for i in range(1, 11):
    _inv[i] = pow(i, -1, MOD)

# Generate all non-empty subsets with their size and sum
# _subset_info[k] = list of (sum_mod_101, tuple_of_denoms)
_subset_info = {k: [] for k in range(1, 11)}
for k in range(1, 11):
    for combo in combinations(DENOMS, k):
        s = 0
        for d in combo:
            s = (s + _inv[d]) % MOD
        _subset_info[k].append((s, combo))

# Find minimum K such that all residues are covered by subsets of size <= K
_best_K = 10
for K in range(1, 11):
    covered = [False] * MOD
    for k in range(1, K + 1):
        for s, _ in _subset_info[k]:
            covered[s] = True
    if all(covered):
        _best_K = K
        break

# Precompute the best (smallest) subset for each residue
# We iterate k from 1 to _best_K, so the first assignment is the smallest size.
_best_for_residue = [None] * MOD
for k in range(1, _best_K + 1):
    for s, combo in _subset_info[k]:
        if _best_for_residue[s] is None:
            _best_for_residue[s] = list(combo)

# Safety check: ensure all residues are covered
# If any is None, it means our logic for _best_K failed, which shouldn't happen.
# But we handle it by falling back to any subset if needed.
for r in range(MOD):
    if _best_for_residue[r] is None:
        for k in range(1, 11):
            for s, combo in _subset_info[k]:
                if s == r:
                    _best_for_residue[r] = list(combo)
                    break
            if _best_for_residue[r] is not None:
                break

# Verify validity
def _verify():
    for r in range(MOD):
        lst = _best_for_residue[r]
        if not lst:
            return False
        if len(lst) != len(set(lst)):
            return False
        s = sum(_inv[d] for d in lst) % MOD
        if s != r:
            return False
    return True

# _verify() should be True

def construct():
    # Return a deep copy to prevent mutation of the global cache
    return [list(lst) for lst in _best_for_residue]
# EVOLVE-BLOCK-END
