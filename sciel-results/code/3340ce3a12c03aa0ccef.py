"""Direct construction of a size-efficient sorting network on 18 wires using Max-SAT-inspired stochastic local search."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Pre-compute bit masks for verification
    total_inputs = 1 << N  # 262144
    NUM_WORDS = total_inputs // 64  # 4096

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism with early termination.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def count_violations(comparators):
        """
        Count total number of unsatisfied sorting constraints (Max-SAT soft objective).
        Returns 0 if network is valid.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        total_violations = 0
        for k in range(N - 1):
            # Count bits where wire[k] has 1 but wire[k+1] has 0
            violations = wires[k] & ~wires[k + 1]
            total_violations += int(np.count_nonzero(violations))
        return total_violations

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            if n < 30:
                positions = list(range(n))
                random.shuffle(positions)
            else:
                positions = random.sample(range(n), min(50, n))

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def random_dense_network(num_comparators=200):
        """Generate a random dense network and shuffle."""
        comparators = []
        for _ in range(num_comparators):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        random.shuffle(comparators)
        return comparators

    def maxsat_local_search(initial_net, t_max=1000, t_min=1.0, alpha=0.998, max_iter=30000, deadline=None):
        """
        Max-SAT-inspired stochastic local search.
        Objective: minimize (comparator_count + violation_penalty * violations)
        Violations act as hard constraints via large penalty.
        """
        current = list(initial_net)
        current_violations = count_violations(current)
        current_energy = len(current) + 1000.0 * current_violations
        
        best = list(current)
        best_energy = current_energy
        best_violations = current_violations
        
        t = t_max
        no_improve_count = 0
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            r = random.random()
            candidate = list(current)
            valid_mutation = True
            
            if r < 0.45:
                # Remove comparator
                if len(candidate) == 0:
                    valid_mutation = False
                else:
                    idx = random.randint(0, len(candidate) - 1)
                    candidate.pop(idx)
            elif r < 0.65:
                # Insert comparator
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    valid_mutation = False
                else:
                    if u > v:
                        u, v = v, u
                    pos = random.randint(0, len(candidate))
                    candidate.insert(pos, (u, v))
            elif r < 0.80:
                # Swap two comparators
                if len(candidate) < 2:
                    valid_mutation = False
                else:
                    i1, i2 = random.sample(range(len(candidate)), 2)
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            elif r < 0.92:
                # Replace a comparator
                if len(candidate) == 0:
                    valid_mutation = False
                else:
                    idx = random.randint(0, len(candidate) - 1)
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v:
                        valid_mutation = False
                    else:
                        if u > v:
                            u, v = v, u
                        candidate[idx] = (u, v)
            else:
                # Move a comparator to a new position
                if len(candidate) < 2:
                    valid_mutation = False
                else:
                    idx = random.randint(0, len(candidate) - 1)
                    comp = candidate.pop(idx)
                    new_pos = random.randint(0, len(candidate))
                    candidate.insert(new_pos, comp)

            if not valid_mutation:
                continue

            new_violations = count_violations(candidate)
            new_energy = len(candidate) + 1000.0 * new_violations
            
            # Acceptance: always accept if better, otherwise with probability exp(-delta/t)
            if new_energy < current_energy:
                current = candidate
                current_energy = new_energy
                current_violations = new_violations
                no_improve_count = 0
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
                    best_violations = new_violations
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    prob = np.exp(-delta / t)
                    if random.random() < prob:
                        current = candidate
                        current_energy = new_energy
                        current_violations = new_violations
                no_improve_count += 1
            
            t *= alpha
            if t < t_min:
                # Restart with higher temperature if stuck
                if no_improve_count > 500:
                    t = t_max * 0.5
                    no_improve_count = 0
                else:
                    break
        
        # Return best valid solution (0 violations) or best overall
        if best_violations == 0:
            return best
        # If best has violations, try to find any valid solution
        # Fall back to current if it's valid
        if current_violations == 0:
            return current
        return best

    def polish_pass(network, deadline):
        """
        Final high-precision polish: aggressive greedy removal with full verification.
        """
        current = list(network)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(current)
            if n == 0:
                break
            # Try all positions in random order
            positions = list(range(n))
            random.shuffle(positions)
            for idx in positions:
                if time.time() > deadline:
                    break
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break
        return current

    # Generate diverse starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    
    # Add random dense starts with different seeds
    for i in range(4):
        random.seed(200 + i * 17)
        candidates.append((f"random_{i}", random_dense_network(120 + i * 20)))

    best_network = None
    best_count = float('inf')

    # Phase 1: Quick prune each candidate
    for name, net in candidates:
        if time.time() > deadline - 30.0:
            break
        pruned = greedy_remove(net, time_limit=min(deadline - 25.0, start_time + 3.0))
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

    # Phase 2: Max-SAT local search from best pruned candidate
    if best_network is not None and time.time() < deadline - 15.0:
        msat_result = maxsat_local_search(
            best_network,
            t_max=500,
            t_min=0.5,
            alpha=0.998,
            max_iter=50000,
            deadline=deadline - 10.0
        )
        if len(msat_result) < best_count:
            best_count = len(msat_result)
            best_network = list(msat_result)

    # Phase 3: Additional SA runs from different starting points if time permits
    if time.time() < deadline - 20.0:
        for name, net in candidates[:3]:
            if time.time() > deadline - 10.0:
                break
            pruned = greedy_remove(net, time_limit=min(deadline - 8.0, start_time + 2.0))
            sa_result = maxsat_local_search(
                pruned,
                t_max=300,
                t_min=0.3,
                alpha=0.9985,
                max_iter=30000,
                deadline=deadline - 8.0
            )
            if len(sa_result) < best_count:
                best_count = len(sa_result)
                best_network = list(sa_result)

    # Phase 4: Final polish pass
    if best_network is not None and time.time() < deadline - 5.0:
        polished = polish_pass(best_network, deadline - 3.0)
        if len(polished) < best_count:
            best_count = len(polished)
            best_network = list(polished)

    # Fallback if something went wrong
    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
