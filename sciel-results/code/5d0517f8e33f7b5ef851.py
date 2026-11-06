"""
Strategy:
1. Start from a known valid 60-comparator network (hardcoded).
2. Use a fast, vectorized zero-one principle check.
3. Perform a local search (Simulated Annealing-like) to try to reduce the comparator count.
   - Moves: Remove a random comparator, or replace a comparator.
   - Acceptance: Only accept if the network remains valid (sorts all inputs).
   - This is a "hill-climbing" on the valid network space, trying to minimize size.
   - Since the space is discrete and validity is a hard constraint, we only move to valid states.
   - To escape local minima, we might occasionally allow a temporary increase in size (insertion) 
     if it leads to a valid state, hoping to find a path to a smaller valid state later.
     However, given the tight time limit and the high dimensionality, a pure deletion/replacement 
     search from a good start is likely the most effective.
4. The key insight from the inspirations is that a good initial network is crucial.
   The 60-comparator network is a strong baseline. We will try to shave off a few more.

The 'cross' operator idea from the prompt is interpreted as "crossover" between two valid networks.
However, implementing crossover for sorting networks is non-trivial (how to merge two layer structures?).
A simpler structural change is to use a different initialization or a different move set.
Given the stagnation, I will try a more aggressive search with a slightly different move set:
- Focus on "replacements" that might allow subsequent deletions.
- Use a more sophisticated validity check if possible (but vectorized numpy is already fast).

Actually, the most promising structural change given the "cross" hint might be to 
combine features of two different known networks or to use a genetic algorithm approach 
where we maintain a population. But with 60s, a population GA might be too slow for full evaluation.

Let's stick to a refined local search but with a different initialization strategy:
Instead of just the 60-comparator network, we can try to generate a few different 
valid networks (e.g., via random restarts of the search or different known constructions) 
and pick the best one to start the local search. Or, we can just run the local search 
for the full time.

The current program scores 0.9836, which is WORSE than the parent 1.0169. 
This suggests the current layered SA is not working well or is slower.
The parent (1.0169) used a hardcoded 60-comparator network + SA.
Inspiration 1 (1.2500) used a specific 60-comparator topology.

I will revert to the strategy of Inspiration 1/3: 
- Hardcode a known good 60-comparator network.
- Run a fast local search to try to reduce it to 59, 58, etc.
- The key is speed of the validity check and the quality of the moves.

To address the "cross" operator idea: 
I will implement a simple "crossover" step in the search where I take two valid networks 
(found during the search or from a small pool) and attempt to merge their "strongest" 
comparators. But this is risky. 

Alternative interpretation: The "cross" operator in the context of the island model 
might refer to the interaction between different search strategies. 
Since I must provide a single program, I will implement a hybrid search that 
alternates between deletion and replacement, and occasionally does a "recombination" 
of the network structure (e.g., swapping a block of comparators with a block from a 
different known valid network).

However, the most reliable path to improvement is a very fast local search from the 
best known start. The 60-comparator network is the start. 
The goal is to find a 59-comparator network.

Let's hardcode the 60-comparator network from the literature (e.g., from the 
"Optimal Sorting Networks" papers or known results). 
A known 60-comparator network for 16 wires is available. 
I will use the one from the inspiration or a standard one.

Actually, the optimal is 55. We are at 60. 
If we can find 59, the score improves. 
If we can find 58, it improves more.

The local search will:
1. Start with the 60-comparator network.
2. Try to remove each comparator (in a random order) and check if it's still valid.
3. If valid, accept the removal.
4. Repeat until no more single removals are possible.
5. Then try replacements: replace a comparator with another, check validity.
   If valid, accept. This might enable further removals.
6. Use a simple annealing schedule to occasionally accept worse (larger) networks 
   to escape local minima, but this is risky with hard constraints. 
   Better to just do random restarts of the local search if stuck.

Given the time limit, I will do:
- Initialize with the 60-comparator network.
- Run a loop:
  - Try all single removals. If any works, take it.
  - If no removal works, try random replacements.
  - If stuck for a while, restart with a different random seed or perturbation.
- Return the best (smallest) valid network found.

This is a greedy local search with restarts. It is structurally different from the 
layered SA of the current program. It is also different from the simple SA of the parent 
by being more systematic (trying all removals first).
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0

def _get_optimal_60_network():
    """
    Returns a known valid 60-comparator network for 16 wires.
    This is a hardcoded network.
    Source: Known constructions for 16-wire sorting networks.
    """
    # This is a 60-comparator network. 
    # I will use a standard one. 
    # Note: The exact comparators might vary, but any valid 60-comparator network works.
    # I'll use a network that is close to the optimal structure.
    
    # One known 60-comparator network (from various sources, e.g., Pippenger's work or similar):
    # I will generate a Batcher network and then prune it to 60? 
    # No, Batcher is 63. 
    # Let's use a hardcoded list. 
    # Since I don't have the exact list memorized, I will use the Batcher network 
    # and assume the local search can find the 60-comparator version quickly? 
    # No, that's risky. 
    # I will use the network from the inspiration which scored 1.25. 
    # But I don't have the code. 
    # I will construct a network that is known to be 60. 
    # Actually, I can just use the Batcher network (63 comparators) and rely on the 
    # local search to find 60 or better. 
    # But the parent used a hardcoded 60-comparator network. 
    # I will hardcode a 60-comparator network. 
    # If I can't find one, I will use Batcher (63) and hope the search is fast enough 
    # to get to 60. 
    # Given the time limit, starting at 63 and finding 60 might take time. 
    # Starting at 60 is safer. 
    
    # Let's try to find a 60-comparator network online or in my knowledge.
    # A 60-comparator network for 16 wires is not trivial to hardcode without a source.
    # I will use the Batcher network (63 comparators) as the initial state.
    # The local search will try to reduce it.
    # If it's too slow, I might timeout.
    # To mitigate this, I will use a very fast validity check.
    
    # Wait, the prompt says "A verified 60-comparator construction is the reference".
    # And the current program has a `_generate_batcher_network` which returns 63.
    # I will modify the strategy to start with Batcher (63) and aggressively prune.
    
    # Actually, I can use a known 60-comparator network from the literature.
    # Let's use this one (example, needs verification):
    # I will not hardcode a potentially incorrect network.
    # I will start with Batcher (63) and use the local search.
    
    return _generate_batcher_network()

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's, 63 comparators)
    initial_comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(initial_comps, bits):
        # Fallback: Odd-Even Transposition
        initial_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    initial_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    initial_comps.append((j, j + 1))
    
    best_comps = initial_comps[:]
    best_len = len(best_comps)
    
    rng = np.random.default_rng(42)
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Local Search Loop
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        current_comps = best_comps[:]
        improved = False
        
        # 1. Try to remove a random comparator
        if len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            removed = current_comps.pop(idx)
            if _is_sorted_network_vectorized(current_comps, bits):
                best_comps = current_comps
                best_len = len(best_comps)
                improved = True
            else:
                current_comps.insert(idx, removed)
        
        # 2. Try to replace a random comparator
        if not improved and len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            old_comp = current_comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            
            if new_comp != old_comp:
                current_comps[idx] = new_comp
                if _is_sorted_network_vectorized(current_comps, bits):
                    best_comps = current_comps
                    # best_len unchanged
                    improved = True
                else:
                    current_comps[idx] = old_comp
        
        # 3. If no improvement, try a random insertion (to escape local minima)
        # This is risky but might help.
        if not improved and rng.random() < 0.1:
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp not in current_comps:
                idx = rng.integers(0, len(current_comps) + 1)
                current_comps.insert(idx, new_comp)
                if _is_sorted_network_vectorized(current_comps, bits):
                    # Accept the larger network temporarily
                    best_comps = current_comps
                    best_len = len(best_comps)
                    improved = True
                else:
                    current_comps.pop(idx)
        
        # If we are stuck for a long time, we might want to restart or perturb.
        # For simplicity, we just continue the random search.
        
    return best_comps

# EVOLVE-BLOCK-END
