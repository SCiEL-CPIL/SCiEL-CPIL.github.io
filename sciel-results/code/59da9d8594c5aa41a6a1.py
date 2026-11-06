"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Precompute all possible (i, j) pairs with i < j
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_pairs = len(all_pairs)
    pair_to_idx = {p: k for k, p in enumerate(all_pairs)}
    
    # Precompute the effect of each comparator on all 2^N boolean inputs
    # For each input mask (0 to 2^N - 1), and each comparator (i,j),
    # we compute the resulting mask after applying the comparator.
    # But 2^18 * 153 is too large to precompute all. Instead, we'll use
    # a different approach: simulate the network on random/test inputs.
    
    # Strategy: Start with a known-correct network (odd-even transposition),
    # then use local search to remove/swap/move comparators while maintaining correctness.
    
    # Odd-even transposition network
    def odd_even_transposition():
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Verification: check if network sorts all boolean inputs
    # Using the 0-1 principle, we only need to check 2^N boolean inputs.
    # For efficiency, we use a two-stage verification:
    # Stage 1: Check a random subset of inputs (fast)
    # Stage 2: Check all 2^N inputs (slow, only if stage 1 passes)
    
    def apply_network(comps, input_mask):
        """Apply the network to a boolean input mask, return output mask."""
        wires = [(input_mask >> i) & 1 for i in range(N)]
        for (i, j) in comps:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        result = 0
        for i in range(N):
            result |= wires[i] << i
        return result
    
    def is_sorted_mask(mask):
        """Check if a boolean mask represents a sorted sequence (all 0s then all 1s)."""
        # A sorted boolean sequence has the form 000...0111...1
        # Find the position where it switches from 0 to 1
        first_one = -1
        for i in range(N):
            if (mask >> i) & 1:
                first_one = i
                break
        if first_one == -1:
            return True  # All zeros
        for i in range(first_one):
            if (mask >> i) & 1:
                return False
        for i in range(first_one, N):
            if not ((mask >> i) & 1):
                return False
        return True
    
    def verify_full(comps):
        """Verify network sorts all 2^N boolean inputs. Returns True if correct."""
        for mask in range(1 << N):
            out = apply_network(comps, mask)
            if not is_sorted_mask(out):
                return False
        return True
    
    def verify_subset(comps, num_samples=500, rng=None):
        """Quick verification on a subset of inputs."""
        if rng is None:
            rng = random.Random(42)
        # Test all masks with small number of 1s (these are the hardest)
        # and some random ones
        test_masks = set()
        # All masks with k ones for k in 0..N
        for k in range(N + 1):
            # Sample some masks with exactly k ones
            if k == 0:
                test_masks.add(0)
            elif k == N:
                test_masks.add((1 << N) - 1)
            else:
                for _ in range(min(20, (1 << N) // k)):
                    ones = rng.sample(range(N), k)
                    mask = 0
                    for pos in ones:
                        mask |= (1 << pos)
                    test_masks.add(mask)
        
        for mask in test_masks:
            out = apply_network(comps, mask)
            if not is_sorted_mask(out):
                return False
        return True
    
    # Start with odd-even transposition
    current = odd_even_transposition()
    best = list(current)
    best_len = len(best)
    
    # Local search: try to remove, swap, or move comparators
    rng = random.Random(12345)
    
    def try_remove(comps, idx):
        """Try removing comparator at index idx."""
        new_comps = comps[:idx] + comps[idx+1:]
        return new_comps
    
    def try_swap(comps, i, j):
        """Try swapping comparators at positions i and j."""
        new_comps = list(comps)
        new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
        return new_comps
    
    def try_move(comps, src, dst):
        """Try moving comparator from src to dst."""
        new_comps = list(comps)
        item = new_comps.pop(src)
        new_comps.insert(dst, item)
        return new_comps
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        step += 1
        n = len(current)
        if n <= best_len:
            # Try to remove a comparator
            if n > 0:
                idx = rng.randrange(n)
                new_comps = try_remove(current, idx)
                if verify_subset(new_comps, rng=rng):
                    # Passed subset check, do full verification
                    if verify_full(new_comps):
                        current = new_comps
                        best = list(current)
                        best_len = len(current)
                        no_improve_steps = 0
                        continue
            # Try swapping two comparators
            if n >= 2:
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i != j:
                    new_comps = try_swap(current, i, j)
                    if verify_subset(new_comps, rng=rng):
                        if verify_full(new_comps):
                            current = new_comps
                            no_improve_steps = 0
                            continue
            # Try moving a comparator
            if n >= 2:
                src = rng.randrange(n)
                dst = rng.randrange(n)
                if src != dst:
                    new_comps = try_move(current, src, dst)
                    if verify_subset(new_comps, rng=rng):
                        if verify_full(new_comps):
                            current = new_comps
                            no_improve_steps = 0
                            continue
        
        no_improve_steps += 1
        
        # If we're stuck, try a random perturbation
        if no_improve_steps % 1000 == 0:
            # Add a random comparator to escape local minimum
            if len(current) < MAX_COMPARATORS:
                pair = rng.choice(all_pairs)
                pos = rng.randrange(len(current) + 1)
                new_comps = current[:pos] + [pair] + current[pos:]
                if verify_subset(new_comps, rng=rng):
                    current = new_comps
    
    return best
# EVOLVE-BLOCK-END
