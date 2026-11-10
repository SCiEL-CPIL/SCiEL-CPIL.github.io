import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    timeout = 115  # Safety margin for 150s wall-clock limit

    def build_counts_numpy(chosen):
        """Build sum-count array using numpy convolution."""
        if len(chosen) == 0:
            return np.zeros(max_sum + 1, dtype=np.int32)
        arr = np.zeros(N + 1, dtype=np.int32)
        for x in chosen:
            arr[x] = 1
        # Convolution gives counts of a+b where a,b in set (order matters)
        conv = np.convolve(arr, arr, mode='full')
        # conv[s] = number of ordered pairs (a,b) with a+b=s
        # We need unordered pairs with a<=b
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for s in range(2, max_sum + 1):
            ordered = conv[s] if s < len(conv) else 0
            if s % 2 == 0 and s // 2 <= N and arr[s // 2] == 1:
                counts[s] = (ordered - 1) // 2 + 1
            else:
                counts[s] = ordered // 2
        return counts

    class B2State:
        """Efficient state management for B2[2] constraint."""
        def __init__(self):
            self.chosen = []
            self.chosen_arr = np.zeros(N + 1, dtype=np.int32)
            self.counts = np.zeros(max_sum + 1, dtype=np.int32)
            self.full_sums = set()  # Sums with count == 2
        
        def can_add(self, x):
            """Check if x can be added to the set - vectorized."""
            if self.chosen_arr[x]:
                return False
            if self.counts[2 * x] >= 2:
                return False
            ys = np.where(self.chosen_arr)[0]
            if len(ys) == 0:
                return True
            sums = x + ys
            # Vectorized check: any sum already full?
            valid_sums = sums[(sums >= 2) & (sums <= max_sum)]
            if len(valid_sums) > 0:
                return not np.any(self.counts[valid_sums] >= 2)
            return True
        
        def add(self, x):
            """Add x to the set."""
            for y in self.chosen:
                s = x + y
                self.counts[s] += 1
                if self.counts[s] == 2:
                    self.full_sums.add(s)
            s2 = 2 * x
            self.counts[s2] += 1
            if self.counts[s2] == 2:
                self.full_sums.add(s2)
            self.chosen.append(x)
            self.chosen_arr[x] = 1
        
        def remove(self, x):
            """Remove x from the set."""
            idx = self.chosen.index(x)
            for i, y in enumerate(self.chosen):
                if y == x:
                    continue
                s = x + y
                self.counts[s] -= 1
                if self.counts[s] == 1:
                    self.full_sums.discard(s)
            s2 = 2 * x
            self.counts[s2] -= 1
            if self.counts[s2] == 1:
                self.full_sums.discard(s2)
            self.chosen.pop(idx)
            self.chosen_arr[x] = 0
        
        def size(self):
            return len(self.chosen)

    def greedy_construct(rng, strategy='random'):
        """Greedy with different strategies using efficient state."""
        state = B2State()
        
        if strategy == 'gap':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif strategy == 'ascending':
            candidates = list(range(1, N + 1))
        elif strategy == 'descending':
            candidates = list(range(N, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif strategy == 'sparse':
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        elif strategy == 'prime':
            # Use primes as a heuristic for sparse sets
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in range(1, N + 1) if is_prime(x)]
            others = [x for x in range(1, N + 1) if not is_prime(x)]
            rng.shuffle(primes)
            rng.shuffle(others)
            candidates = primes + others
        elif strategy == 'sqrt':
            # Elements near sqrt(k) for k in 1..N
            candidates = sorted(set([int(np.sqrt(k)) for k in range(1, N + 1)]))
            candidates = [x for x in candidates if 1 <= x <= N]
            rng.shuffle(candidates)
        elif strategy == 'conflict_min':
            # Start with numbers that have fewer potential conflicts
            # Larger numbers have fewer pairs (a,b) with a+b = x where both a,b <= N
            candidates = list(range(N, 0, -1))
            rng.shuffle(candidates)
        elif strategy == 'mixed':
            # Mix of small and large numbers
            small = list(range(1, N // 4 + 1))
            large = list(range(3 * N // 4, N + 1))
            middle = list(range(N // 4 + 1, 3 * N // 4))
            rng.shuffle(small)
            rng.shuffle(large)
            rng.shuffle(middle)
            # Interleave
            candidates = []
            si, li, mi = 0, 0, 0
            while si < len(small) or li < len(large) or mi < len(middle):
                if si < len(small):
                    candidates.append(small[si]); si += 1
                if li < len(large):
                    candidates.append(large[li]); li += 1
                if mi < len(middle):
                    candidates.append(middle[mi]); mi += 1
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)

        for x in candidates:
            if state.can_add(x):
                state.add(x)

        return state

    def local_search(state, rng, max_time=30):
        """Local search: try to add elements, then try swaps."""
        best_chosen = state.chosen[:]
        best_len = state.size()
        deadline = start_time + max_time
        stuck_count = 0

        while time.time() < deadline:
            improved = False

            # Phase 1: Try to add any missing element (try all, not just 100)
            missing = [x for x in range(1, N + 1) if not state.chosen_arr[x]]
            if missing:
                rng.shuffle(missing)
                for x in missing:
                    if state.can_add(x):
                        state.add(x)
                        improved = True
                        if state.size() > best_len:
                            best_len = state.size()
                            best_chosen = state.chosen[:]
                        break

            if improved:
                stuck_count = 0
                continue

            # Phase 2: Try swaps - remove one, add another (more aggressive)
            if state.size() > 0 and len(missing) > 0:
                # Try more removal candidates
                sample_size = min(state.size(), 50)
                to_try_remove = random.sample(state.chosen, sample_size)

                for x in to_try_remove:
                    # Temporarily remove x
                    state.remove(x)

                    # Try to find something to add (try more candidates)
                    missing_now = [y for y in range(1, N + 1) if not state.chosen_arr[y]]
                    if missing_now:
                        rng.shuffle(missing_now)
                        found = False
                        for y in missing_now[:200]:
                            if state.can_add(y):
                                state.add(y)
                                improved = True
                                found = True
                                if state.size() > best_len:
                                    best_len = state.size()
                                    best_chosen = state.chosen[:]
                                break

                    if not found:
                        # Put x back
                        state.add(x)

                    if improved:
                        stuck_count = 0
                        break

            if not improved:
                stuck_count += 1
                if stuck_count >= 5:
                    break

        return best_chosen

    def verify(chosen):
        """Verify B2[2] constraint."""
        counts = {}
        for i, a in enumerate(chosen):
            for b in chosen[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return all(v <= 2 for v in counts.values())

    best_set = []
    best_len = 0

    # Multiple greedy starts with different strategies
    strategies = ['random', 'gap', 'sparse', 'ascending', 'descending', 'prime', 'sqrt', 'conflict_min', 'mixed']
    rng = random.Random(42)

    # First pass: run all strategies
    for strat in strategies:
        if time.time() - start_time > timeout - 50:
            break
        state = greedy_construct(rng, strat)
        if state.size() > best_len:
            best_len = state.size()
            best_set = state.chosen[:]

    # Local search on best
    if best_set and time.time() - start_time < timeout - 30:
        # Rebuild state from best_set
        state = B2State()
        for x in best_set:
            state.add(x)
        remaining = min(40, timeout - 30 - (time.time() - start_time))
        if remaining > 5:
            improved_chosen = local_search(state, rng, max_time=remaining)
            if len(improved_chosen) > best_len:
                best_len = len(improved_chosen)
                best_set = improved_chosen[:]

    # More restarts if time allows - use diverse strategies
    all_strats = ['random', 'gap', 'sparse', 'prime', 'sqrt', 'conflict_min', 'mixed']
    restart_count = 0
    while time.time() - start_time < timeout - 10:
        strat = all_strats[restart_count % len(all_strats)]
        state = greedy_construct(random.Random(), strat)
        if state.size() > best_len:
            best_len = state.size()
            best_set = state.chosen[:]
            # Quick local search
            remaining = min(20, timeout - 10 - (time.time() - start_time))
            if remaining > 3:
                improved_chosen = local_search(state, random.Random(), max_time=remaining)
                if len(improved_chosen) > best_len:
                    best_len = len(improved_chosen)
                    best_set = improved_chosen[:]
        restart_count += 1

    # Final verification
    if best_set and not verify(best_set):
        # Fallback: rebuild from scratch with simpler approach
        chosen = []
        counts = {}
        for x in range(1, N + 1):
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    valid = False
            if valid:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen.append(x)

        best_set = chosen
        best_len = len(chosen)

    return best_set
