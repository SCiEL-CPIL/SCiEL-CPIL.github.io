import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin before 120s limit

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute K(2,2,2) configurations using numpy for speed
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    def get_pair_partitions(six):
        """Generate all 15 ways to partition 6 vertices into 3 pairs."""
        v = list(six)
        parts = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (rem[0], rem[j])
                if j == 1:
                    pair3 = (rem[2], rem[3])
                elif j == 2:
                    pair3 = (rem[1], rem[3])
                else:
                    pair3 = (rem[1], rem[2])
                parts.append([pair1, pair2, pair3])
        return parts

    for six_subset in combinations(range(n), 6):
        for partition in get_pair_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)

    num_k222s = len(k222_triples)  # 120120

    # Count how many K(2,2,2)s each triple is in
    triple_k222_count = np.array([len(triple_to_k222s[t]) for t in range(num_triples)], dtype=np.int32)

    class State:
        def __init__(self):
            self.present = np.zeros(num_triples, dtype=bool)
            self.k_counts = np.zeros(num_k222s, dtype=np.int32)
            self.size = 0
            self.missing = set(range(num_triples))

        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present[t_id] = True
            self.size += 1
            self.missing.discard(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            self.missing.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1

        def present_list(self):
            return [i for i in range(num_triples) if self.present[i]]

        def copy(self):
            s = State()
            s.present = self.present.copy()
            s.k_counts = self.k_counts.copy()
            s.size = self.size
            s.missing = set(self.missing)
            return s

    def polish(state, max_iters=200):
        """Try to add any missing triple that fits, iterating until no improvement."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Try to add from missing set
            missing_list = list(state.missing)
            random.shuffle(missing_list)
            for t_id in missing_list:
                if time.time() > DEADLINE:
                    break
                if state.add(t_id):
                    improved = True

    def simulated_annealing(state, steps=5000, t_start=1.0, t_end=0.01):
        """Simulated annealing to explore the solution space."""
        current_size = state.size
        best_size = state.size
        best_state = state.copy()
        temperature = t_start
        cooling = (t_start / max(t_end, 1e-6)) ** (1.0 / max(steps, 1))

        for step in range(steps):
            if time.time() > DEADLINE:
                break
            temperature *= cooling

            # Choose a random move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.4:
                # Try to add a missing triple
                if not state.missing:
                    continue
                t_id = random.choice(list(state.missing))
                if state.add(t_id):
                    delta = 1
                else:
                    delta = 0
                    continue
            elif move_type < 0.7:
                # Try to remove a present triple
                present_list = state.present_list()
                if not present_list:
                    continue
                t_id = random.choice(present_list)
                state.remove(t_id)
                delta = -1
            else:
                # Swap: remove one, add one
                present_list = state.present_list()
                if not present_list or not state.missing:
                    continue
                t_out = random.choice(present_list)
                t_in = random.choice(list(state.missing))
                state.remove(t_out)
                if state.add(t_in):
                    delta = 0
                else:
                    # Revert
                    state.add(t_out)
                    continue

            new_size = state.size
            if new_size > best_size:
                best_size = new_size
                best_state = state.copy()

            # Accept or reject based on temperature
            if delta >= 0 or random.random() < np.exp(-abs(delta) / max(temperature, 1e-6)):
                pass  # Accept
            else:
                # Revert
                if move_type < 0.4:
                    state.remove(t_id)
                elif move_type < 0.7:
                    state.add(t_id)
                else:
                    state.remove(t_in)
                    state.add(t_out)

        # Restore best state
        state.present = best_state.present.copy()
        state.k_counts = best_state.k_counts.copy()
        state.size = best_state.size
        state.missing = set(best_state.missing)
        return best_size

    best_size = 0
    best_present = None

    # Greedy initialization
    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        # Sort by number of K(2,2,2)s (fewer constraints first)
        order = np.argsort(triple_k222_count)
        # Add randomness to the ordering
        for i in range(len(order)):
            if i > 0 and i < len(order):
                swap_idx = i - local_rng.randint(0, min(i, 15))
                order[i], order[swap_idx] = order[swap_idx], order[i]
        for t_id in order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state

    # Main loop: multiple restarts with simulated annealing
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919

        if restart_count <= 3:
            # Greedy init
            state = greedy_init(seed)
        else:
            # Start from best and perturb
            state = State()
            if best_present is not None:
                state.present = best_present.copy()
                state.size = int(state.present.sum())
                for k_idx in range(num_k222s):
                    count = 0
                    for t_id in k222_triples[k_idx]:
                        if state.present[t_id]:
                            count += 1
                    state.k_counts[k_idx] = count
                state.missing = set(i for i in range(num_triples) if not state.present[i])

            # Perturb: remove random triples
            n_perturb = min(30, state.size // 6)
            for _ in range(n_perturb):
                if time.time() > DEADLINE:
                    break
                present_list = state.present_list()
                if present_list:
                    t = random.choice(present_list)
                    state.remove(t)
            polish(state)

        # Simulated annealing
        sa_steps = min(5000, max(1000, int((DEADLINE - time.time()) * 50)))
        simulated_annealing(state, steps=sa_steps)

        # Final polish
        polish(state, max_iters=100)

        if state.size > best_size:
            best_size = state.size
            best_present = state.present.copy()

    return [list(triples[t_id]) for t_id in range(num_triples) if best_present is not None and best_present[t_id]]
