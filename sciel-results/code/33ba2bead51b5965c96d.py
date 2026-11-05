import numpy as np
import time
import random

N = 18
SIZE = 1 << N  # 262144

def construct():
    """
    Constructs a sorting network using a high-speed Boolean Matrix simulation
    to enable aggressive greedy comparator removal and local search.
    
    The state is maintained as a 2D array `perm` of shape (N, SIZE) where 
    perm[i, k] indicates if wire i holds the bit from input k's i-th position 
    after the current prefix of the network. 
    Actually, for binary sorting, we track the value. 
    Let's stick to the value tracking but optimize the check.
    
    Wait, the previous approach tracked values `states[i]` for all k.
    states[i] is a vector of length SIZE.
    Comparator (i, j):
    min_i = states[i] & states[j]
    max_j = states[i] | states[j]
    
    This is already O(SIZE) per comparator.
    The bottleneck in the previous code was likely the Python loop overhead 
    and the `copy()` calls or the `_is_sorted` check which is O(N*SIZE).
    
    New Strategy:
    1. Use a single 2D array `states` (N, SIZE) of uint8.
    2. Optimization: Instead of checking sortedness by comparing adjacent rows 
       (which is N-1 vector comparisons), we can maintain a "disorder" counter 
       or simply rely on the fact that if we remove a comparator and the network 
       was sorted, it remains sorted only if that comparator was redundant.
       
    Actually, the key insight from "Targeted replacement... end-biased greedy removal"
    is that we should focus on removing comparators from the end.
    
    To speed up `_is_sorted`:
    `states[:-1] < states[1:]` creates a temporary array of size (N-1, SIZE).
    This is expensive.
    
    Alternative:
    We can check sortedness by verifying that for each input k, the bits are sorted.
    But doing this per-input is slow.
    
    Let's try a different angle: **Reverse Greedy with Caching**.
    
    However, the most significant speedup comes from **avoiding the full sort check** 
    when possible, or making the sort check extremely fast.
    
    Let's look at the "Boolean Matrix" idea again.
    If we represent the network as a sequence of permutations, we can't easily 
    "undo" a comparator in the matrix sense without inverting it (which is possible 
    for comparators since they are involutions if we consider the swap part, but 
    min/max is not a permutation, it's a many-to-one mapping on values, but on 
    the *wires* it is a permutation of the set of input wires? No, min/max 
    changes the values.
    
    Let's stick to value tracking but optimize the check.
    
    Optimization 1: Use `np.any` on the difference.
    `diffs = states[1:] - states[:-1]` -> sorted if all diffs >= 0.
    For binary: `states[1:] < states[:-1]` means 1 followed by 0.
    
    Optimization 2: **Lazy Verification**.
    When we try to remove a comparator, we don't need to check the *entire* network 
    again if we can prove the comparator was redundant locally? No, that's hard.
    
    Let's try to implement the **end-biased greedy removal** with a highly optimized 
    vectorized check.
    
    Key improvement: The previous code did `states[i].copy()` which is O(SIZE).
    We can avoid copies if we apply, check, and revert using in-place operations 
    carefully, or just accept the copy cost but reduce the number of checks.
    
    Actually, `states[i] & states[j]` creates a new array.
    
    Let's try a **Bit-Parallel Reverse Search**.
    
    Start with a known good network (e.g., Batcher's Odd-Even Merge or a good heuristic).
    Then, iterate from the last comparator to the first.
    For each comparator, try to remove it.
    If the network remains sorted, keep it removed.
    
    To make this fast:
    1. Pre-compute the initial sorted state? No, we need the state *before* the comparator 
       to test removal.
       
    Better approach:
    Maintain the state of the network *after* each prefix.
    This requires storing N * SIZE * num_comparators bits? No, that's too much memory 
    (18 * 262144 * 100 * 1 byte = 468 MB). This is feasible!
    
    If we store the state after every comparator, we can:
    1. Check if comparator k is redundant by comparing state[k-1] and state[k+1]? 
       No, that's not how it works.
       
    Let's stick to the standard greedy but optimize the `_is_sorted` function.
    `np.any(states[:-1] > states[1:])` is the check.
    
    Let's try a **Hybrid Approach**:
    1. Generate a good seed (Batcher's OEN is good, ~70-80 comparators).
    2. Use **Simulated Annealing** or **Random Restart Greedy**.
    3. The key is speed.
    
    Let's implement a highly optimized `is_sorted` and `apply_comparator`.
    
    def is_sorted(states):
        # states is (N, SIZE) uint8
        # Check if states[i] >= states[i+1] for all i
        # This is equivalent to: no 1s above 0s in any column? No, row-major.
        # Row i is wire i.
        # Sorted means for each column k, bits are 1...10...0.
        # So states[i, k] >= states[i+1, k].
        
        # Vectorized:
        return not np.any(states[:-1] < states[1:])
        
    This creates a temporary array of size (N-1, SIZE). 
    For N=18, SIZE=262144, that's 17 * 262144 bytes = 4.4 MB.
    This allocation happens on every check. This is the bottleneck.
    
    **Optimization**: Pre-allocate the difference array?
    Or, use a loop over rows? No, vectorization is faster.
    
    Can we avoid the allocation?
    `np.less(states[:-1], states[1:], out=diff_buf)`
    `np.any(diff_buf)`
    
    Let's try that.
    
    Also, the seed network should be better than Odd-Even Transposition (153 comps).
    Batcher's Odd-Even Merge for N=18:
    N=18 -> 9+9.
    OEN(9) -> 4+5.
    OEN(4) -> 2+2.
    OEN(2) -> 1+1.
    
    Let's generate Batcher's OEN properly.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    random.seed(42)
    
    # 1. Generate a good seed network
    def batcher_oen(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        mid = n // 2
        left = batcher_oen(mid)
        right = batcher_oen(n - mid)
        
        # Offset right
        right = [(i + mid, j + mid) for i, j in right]
        
        # Merge
        merge = []
        # Odd-even merge steps
        # 1. Compare (0, 2), (1, 3), ... (n-2, n) ? No.
        # Batcher's OEN merge:
        # For k in 0..n/2-1:
        #   Compare (2k, 2k+2) if 2k+2 < n
        #   Compare (2k+1, 2k+3) if 2k+3 < n
        # Then recursively merge the even and odd sub-sequences?
        # Actually, the standard recursive definition is:
        # OEN(n) = OEN(n/2) on evens, OEN(n/2) on odds, then Merge(n/2, n/2)
        # Where Merge(a, b) involves comparing (2i, 2i+1) etc?
        
        # Let's use a known correct implementation or a simpler heuristic if OEN is tricky.
        # For N=18, a simple "AKS" or "Shellsort" network might be a better seed.
        # Or just use the transposition network but it's 153.
        # 77 is the target.
        
        # Let's try to generate a "Shellsort" network (Knuth's).
        # It's known to be efficient.
        pass

    # Let's use a simpler, known good seed: 
    # The "Odd-Even Merge" network is complex to code from scratch correctly in 10 lines.
    # Let's use a **Greedy Construction** from scratch?
    # No, start with a valid network.
    
    # Let's use the **Bitonic Sort** network for N=18? 
    # N must be a power of 2 for standard Bitonic. 18 is not.
    # We can use Bitonic for 16 wires and handle the remaining 2?
    # Or use a generic **Shell Sort** network.
    
    # Shell Sort Network (Knuth):
    # Gaps: 2^k - 1? No, Shell's sequence: 1, 4, 13, 40...
    # For N=18, gaps: 1, 4, 13.
    
    def shell_sort_network(n):
        network = []
        gaps = [1, 4, 13] # For n=18
        for gap in reversed(gaps):
            for i in range(gap, n):
                network.append((i - gap, i))
        return network

    # Let's check size of Shell sort for 18:
    # Gap 13: 18-13=5 comps
    # Gap 4: 18-4=14 comps
    # Gap 1: 17 comps
    # Total: 36 comps.
    # Is this a sorting network? 
    # Shell sort network is NOT a sorting network in the general sense (it relies on 
    # specific gap sequences and may not sort all permutations if gaps are not chosen 
    # carefully for the "network" property which requires fixed comparators).
    # Actually, Shell sort *is* a sorting network if the gaps are decreasing and 
    # the comparators are fixed. But it's often larger than optimal.
    # 36 is smaller than 77, so it MUST be wrong. 
    # Why? Because (i-gap, i) with gap=13 compares (0,13), (1,14)...
    # This does not ensure sortedness.
    
    # Let's stick to **Odd-Even Transposition** as the safe seed (153 comps) 
    # and rely on the **optimized greedy removal** to get us down.
    # The previous run got 1.0 (153). 
    # The "inspiration" mentioned 1.545 -> 1.681 (which is better, meaning lower count? 
    # Score is seed/value. Seed=153. 
    # Score 1.0 -> 153 comps.
    # Score 1.681 -> 153/1.681 = 91 comps.
    # So the goal is to get below 91.
    
    current_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            current_network.append((i, i + 1))
    
    # Pre-allocate states
    k_range = np.arange(SIZE, dtype=np.uint64)
    states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    
    # Pre-allocate diff buffer for is_sorted
    diff_buf = np.empty((N - 1, SIZE), dtype=np.uint8)
    
    def is_sorted(states, diff_buf):
        # states[i] < states[i+1]
        np.less(states[:-1], states[1:], out=diff_buf)
        return not np.any(diff_buf)
    
    best_network = current_network[:]
    best_len = len(current_network)
    
    # Greedy Removal: End-biased
    # We iterate from the end to the start.
    # To make this fast, we maintain the state *after* the current prefix.
    # If we remove a comparator, we need to revert the state to *before* that comparator.
    
    # Strategy:
    # 1. Compute the final state of the seed network.
    # 2. Store the state *before* each comparator? 
    #    Memory: 153 * 18 * 262144 bytes = 720 MB. Too much.
    
    # Alternative: 
    # Just do the greedy removal by re-simulating from scratch? 
    # No, that's O(L^2 * SIZE).
    
    # Better: 
    # Keep `states` as the state after the *entire* current network.
    # When we try to remove comparator at index `idx`, we need to know if the network 
    # without it is sorted.
    # This requires re-simulating the prefix up to `idx`, skipping `idx`, and 
    # applying the suffix.
    # This is O(L * SIZE) per attempt.
    # With L=153, SIZE=262144, that's ~40M ops per attempt.
    # 100 attempts = 4B ops. Too slow in Python/NumPy?
    # NumPy is C-speed. 40M byte ops is ~40ms.
    # 100 attempts = 4s.
    # We have 85s. So we can do ~2000 attempts.
    # This is feasible!
    
    def simulate(network, states_init):
        """Simulate network and return final states."""
        # Copy init states
        s = states_init.copy()
        for i, j in network:
            si = s[i].copy()
            sj = s[j].copy()
            s[i] = si & sj
            s[j] = si | sj
        return s
        
    # But we can do better.
    # Let's use the "prefix states" idea but only for the *last* K comparators?
    # No, let's just use the re-simulation approach but optimized.
    
    # Optimization: 
    # We don't need to copy `states_init` every time if we use a stack of states?
    # No, comparators are not invertible in a simple way for the "state" if we 
    # want to branch.
    
    # Let's try a **Local Search** with **Fast Verification**.
    
    # We will maintain `current_network`.
    # We will maintain `current_states` (final state of current_network).
    # We know `current_states` is sorted.
    
    # Move: Remove comparator at `idx`.
    # 1. Re-simulate `current_network` without `idx` to get `new_states`.
    # 2. Check if `new_states` is sorted.
    # 3. If yes, accept. Update `current_network` and `current_states`.
    
    # To speed up re-simulation:
    # We can pre-compute the state after each prefix of `current_network`?
    # If we do that, we can't easily update it when we remove a comparator 
    # (all subsequent prefix states change).
    
    # So, re-simulation is the way.
    
    # Let's implement a fast re-simulation function.
    
    def try_remove(network, idx, init_states):
        """
        Try to remove comparator at idx.
        Returns (new_network, new_states, success)
        """
        # Re-simulate
        s = init_states.copy()
        for k, (i, j) in enumerate(network):
            if k == idx:
                continue
            si = s[i].copy()
            sj = s[j].copy()
            s[i] = si & sj
            s[j] = si | sj
            
        # Check sorted
        if not is_sorted(s, diff_buf):
            return None, None, False
        
        new_network = network[:]
        new_network.pop(idx)
        return new_network, s, True

    # This is O(L * SIZE) per call.
    # L=153, SIZE=262144.
    # 153 * 262144 * 2 (copy+op) ~ 80M ops.
    # In NumPy, this is fast.
    
    # Let's run a greedy loop.
    # We will try to remove from the end to the start.
    # If a removal succeeds, we restart the scan from the end?
    # Or just continue?
    # Restarting from the end is safer because removals can make earlier 
    # comparators redundant.
    
    # However, 80M ops per attempt * 100 attempts = 8B ops.
    # NumPy can do ~100MB/s effective? No, ~10GB/s.
    # 8B byte ops ~ 0.8s.
    # So 100 attempts in 0.8s.
    # We can do ~1000 attempts in 8s.
    # We have 85s.
    # So we can do ~10,000 attempts.
    
    # Let's implement this.
    
    init_states = states.copy() # The very first input state (identity)
    # Wait, `states` currently holds the bits for all inputs.
    # Yes, `states` is the initial state.
    
    current_network = _seed_network() # 153 comps
    current_states = simulate(current_network, init_states)
    
    best_network = current_network[:]
    best_len = len(current_network)
    
    # Greedy Phase
    # We'll do multiple passes.
    for pass_num in range(10):
        if time.time() > deadline - 10:
            break
            
        removed_any = False
        # Scan from end to start
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
                
            new_net, new_states, success = try_remove(current_network, idx, init_states)
            if success:
                current_network = new_net
                current_states = new_states
                removed_any = True
                if len(current_network) < best_len:
                    best_network = current_network[:]
                    best_len = len(current_network)
                # Restart scan from end? 
                # If we removed one, the indices shift.
                # Let's break and restart the for loop for simplicity and correctness.
                # This is O(L^2) in the worst case but L is small.
                break
        
        if not removed_any:
            break
            
    # Local Search Phase
    # If we are stuck, try random swaps or insertions?
    # Insertions are hard to verify efficiently (need to re-simulate).
    # Swaps: Re-simulate.
    
    # Let's try random swaps to escape local minima.
    for i in range(500):
        if time.time() > deadline - 5:
            break
        
        if len(current_network) < 2:
            break
            
        idx1, idx2 = random.sample(range(len(current_network)), 2)
        
        # Swap
        current_network[idx1], current_network[idx2] = current_network[idx2], current_network[idx1]
        
        # Verify
        new_states = simulate(current_network, init_states)
        if is_sorted(new_states, diff_buf):
            # Accept swap
            current_states = new_states
            # Try to remove again
            # (Optional: trigger a quick greedy pass)
            # For now, just keep the swap if it doesn't increase length (it doesn't)
            # But does it help? Maybe it enables future removals.
            pass
        else:
            # Revert
            current_network[idx1], current_network[idx2] = current_network[idx2], current_network[idx1]
            
    # Final Greedy Pass
    if time.time() < deadline - 2:
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            new_net, new_states, success = try_remove(current_network, idx, init_states)
            if success:
                current_network = new_net
                current_states = new_states
                if len(current_network) < best_len:
                    best_network = current_network[:]
                    best_len = len(current_network)
                break # Restart

    return best_network

def _seed_network():
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

# Helper functions defined inside construct or globally?
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program had `_seed_network`, `_is_sorted`, etc.
# I will define them globally to be safe, or inside if they are not in the seed.
# The seed program had:
# _batcher_odd_even_merge
# _seed_network
# _is_sorted
# _apply_comparator
# _restore_comparator
# _verify_network
# construct

# I will keep `_seed_network` and `construct`.
# I will remove the unused ones or keep them if they don't hurt.

def _is_sorted(states):
    """Check if all 2^N inputs are sorted."""
    return not np.any(states[:-1] < states[1:])

def _apply_comparator(states, i, j):
    """Apply comparator (i, j) to states. Returns saved values for restoration."""
    si = states[i].copy()
    sj = states[j].copy()
    states[i] = si & sj
    states[j] = si | sj
    return si, sj

def _restore_comparator(states, i, j, si, sj):
    """Restore states after a comparator was applied."""
    states[i] = si
    states[j] = sj

def _verify_network(network, states):
    """Verify that a network sorts all inputs. Returns True if sorted."""
    for i, j in network:
        si, sj = states[i].copy(), states[j].copy()
        states[i] = si & sj
        states[j] = si | sj
    return _is_sorted(states)

# Note: The `simulate` function used in `construct` is local.
# I should ensure `init_states` is correct.
# `states` in `construct` is initialized as the identity bits.
# `init_states` should be a copy of that.
