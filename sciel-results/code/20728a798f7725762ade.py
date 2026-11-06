import random
import math
import time

def construct():
    targets = set(range(1, 31))
    max_iters = 50000
    timeout = 100.0

    def diff_set(A):
        return {A[j] - A[i] for i in range(len(A)) for j in range(i + 1, len(A))}

    def energy(A):
        if not A or A[0] != 0:
            return float('inf')
        diffs = diff_set(A)
        # collisions: extra occurrences of same diff
        # we count total pairs minus unique diffs
        total_pairs = len(A) * (len(A) - 1) // 2
        collisions = total_pairs - len(diffs)
        missing = len(targets - diffs)
        diameter = A[-1]
        # Prefer size around 9..12; penalize too small or too large
        size_penalty = 0.0
        if len(A) < 9:
            size_penalty += (9 - len(A)) * 50.0
        elif len(A) > 12:
            size_penalty += (len(A) - 12) * 5.0
        return 100.0 * collisions + 50.0 * missing + 1.0 * diameter + size_penalty

    def is_valid(A):
        if not A or A[0] != 0 or len(A) < 9:
            return False
        diffs = diff_set(A)
        if len(targets - diffs) > 0:
            return False
        # distinctness
        total_pairs = len(A) * (len(A) - 1) // 2
        if total_pairs != len(diffs):
            return False
        return True

    def random_init():
        A = [0]
        for _ in range(8):
            A.append(random.randint(1, 60))
        A = sorted(set(A))
        if A[0] != 0:
            A.insert(0, 0)
        return A

    best_solution = None
    best_energy = float('inf')
    start_time = time.time()

    for restart in range(20):
        if time.time() - start_time > timeout:
            break
        A = random_init()
        cur_energy = energy(A)
        T = 80.0
        cooling = 0.997
        for i in range(max_iters):
            if time.time() - start_time > timeout:
                break
            new_A = A[:]
            r = random.random()
            if r < 0.75:
                # perturb one element
                if len(new_A) > 1:
                    idx = random.randint(1, len(new_A) - 1)
                    delta = random.randint(-15, 15)
                    nv = new_A[idx] + delta
                    if nv > 0 and nv not in new_A:
                        new_A[idx] = nv
                        new_A.sort()
                    else:
                        nv = random.randint(1, 200)
                        if nv not in new_A:
                            new_A[idx] = nv
                            new_A.sort()
                        else:
                            continue
            elif r < 0.90:
                # insert
                if len(new_A) < 14:
                    nv = random.randint(1, 200)
                    if nv not in new_A:
                        new_A.append(nv)
                        new_A.sort()
                    else:
                        continue
            else:
                # remove
                if len(new_A) > 9:
                    idx = random.randint(1, len(new_A) - 1)
                    new_A.pop(idx)
            new_energy = energy(new_A)
            dE = new_energy - cur_energy
            if dE < 0 or random.random() < math.exp(-dE / T):
                A = new_A
                cur_energy = new_energy
                if is_valid(A) and cur_energy < best_energy:
                    best_energy = cur_energy
                    best_solution = A[:]
            T *= cooling
        if best_solution:
            break

    if best_solution:
        return best_solution

    # Deterministic fallback: greedy construction ensuring validity
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]

    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        found = False
        # try adding one element
        candidates = sorted({v + s for v in a for s in (d, -d) if v + s >= 0} - set(a))
        for x in candidates:
            trial = sorted(a + [x])
            dl = diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            # add two elements y, y+d
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    return a

if __name__ == "__main__":
    res = construct()
    print(res)
