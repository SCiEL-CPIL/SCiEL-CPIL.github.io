import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 1000
    MAX_SUM = 2 * N

    def calc_violations(A, C):
        violations = 0
        odd_indices = np.arange(1, MAX_SUM + 1, 2)
        if len(odd_indices) > 0:
            violations += np.sum(C[odd_indices] > 4)
        even_indices = np.arange(2, MAX_SUM + 1, 2)
        if len(even_indices) > 0:
            half_indices = even_indices // 2
            has_double = A[half_indices] == 1
            viol1 = np.sum(C[even_indices[has_double]] > 3)
            viol2 = np.sum(C[even_indices[~has_double]] > 4)
            violations += viol1 + viol2
        return violations

    def greedy_init():
        chosen = []
        A = np.zeros(N + 1, dtype=np.int32)
        C = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in range(1, N + 1):
            valid = True
            s = 2 * x
            if s <= MAX_SUM:
                if C[s] > 3:
                    valid = False
            if valid:
                for y in chosen:
                    s = x + y
                    if s <= MAX_SUM:
                        if s % 2 == 1:
                            if C[s] > 2:
                                valid = False
                                break
                        else:
                            if A[s // 2] == 1:
                                if C[s] > 1:
                                    valid = False
                                    break
                            else:
                                if C[s] > 2:
                                    valid = False
                                    break
            if valid:
                chosen.append(x)
                A[x] = 1
                C[2 * x] += 1
                for y in chosen[:-1]:
                    s = x + y
                    C[s] += 2
        return chosen, A, C

    S, A, C = greedy_init()
    best_set = S
    K = len(S)

    def try_increase_k(S_in, A_in, C_in, time_limit):
        S_trial = list(S_in)
        A_trial = A_in.copy()
        C_trial = C_in.copy()
        
        candidates = [i for i in range(1, N + 1) if i not in S_trial]
        if not candidates:
            return None, None, None

        y = random.choice(candidates)
        S_trial.append(y)
        A_trial[y] = 1
        C_trial[2 * y] += 1
        for x in S_in:
            s = x + y
            C_trial[s] += 2

        current_viol = calc_violations(A_trial, C_trial)
        if current_viol == 0:
            return S_trial, A_trial, C_trial

        for step in range(10000):
            if time.time() > time_limit:
                return None, None, None
            if current_viol == 0:
                return S_trial, A_trial, C_trial

            idx = random.randint(0, len(S_trial) - 1)
            x_remove = S_trial[idx]
            y_add = random.randint(1, N)
            if y_add in S_trial:
                continue

            # Remove x_remove
            A_trial[x_remove] = 0
            C_trial[2 * x_remove] -= 1
            for x in S_trial:
                if x == x_remove:
                    continue
                s = x + x_remove
                C_trial[s] -= 2

            # Add y_add
            A_trial[y_add] = 1
            C_trial[2 * y_add] += 1
            for x in S_trial:
                if x == y_add:
                    continue
                s = x + y_add
                C_trial[s] += 2

            S_trial[idx] = y_add
            new_viol = calc_violations(A_trial, C_trial)

            if new_viol <= current_viol:
                current_viol = new_viol
            else:
                # Revert
                A_trial[y_add] = 0
                C_trial[2 * y_add] -= 1
                for x in S_trial:
                    if x == y_add:
                        continue
                    s = x + y_add
                    C_trial[s] -= 2

                A_trial[x_remove] = 1
                C_trial[2 * x_remove] += 1
                for x in S_trial:
                    if x == x_remove:
                        continue
                    s = x + x_remove
                    C_trial[s] += 2

                S_trial[idx] = x_remove

        return None, None, None

    while time.time() < deadline:
        new_S, new_A, new_C = try_increase_k(best_set, A, C, deadline)
        if new_S is not None:
            best_set = new_S
            A = new_A
            C = new_C
            K += 1
        else:
            # If stuck, maybe try to re-shuffle current K to find a better configuration
            # But K is valid, so we just wait or try again with different randomness
            # To avoid infinite loop on same state, we can add a small delay or just break if no progress for a while
            pass

    return sorted(best_set)
