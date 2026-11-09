import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 113.0
    N = 32
    MAX_VAL = 512
    random.seed(42)

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_bit, new_bit):
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, 256):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta

    def gen_candidates():
        candidates = []
        # Consecutive
        candidates.append(list(range(1, 33)))
        # APs
        for step in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                candidates.append(lst)
        # Sumset structures
        for mult in [2, 3, 4, 5]:
            for trial in range(20):
                sizeA = random.randint(3, 8)
                sizeB = random.randint(3, 8)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    candidates.append(lst)
        # Random
        for _ in range(30):
            candidates.append(random.sample(range(1, MAX_VAL + 1), N))
        return candidates

    def local_search(mask, max_iter=300, time_limit=None):
        s = score_mask(mask)
        s_set = set(mask_to_list(mask))
        for _ in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            elems = list(s_set)
            random.shuffle(elems)
            for old in elems:
                old_bit = 1 << (old - 1)
                best_new = None
                best_s = s
                for _ in range(15):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    new_bit = 1 << (new - 1)
                    delta = delta_score(mask, old_bit, new_bit)
                    new_s = s + delta
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new is not None:
                    new_bit = 1 << (best_new - 1)
                    mask = mask ^ old_bit ^ new_bit
                    s_set.discard(old)
                    s_set.add(best_new)
                    s = best_s
                    improved = True
            if not improved:
                break
        return mask, s

    def simulated_annealing(mask, time_limit):
        s = score_mask(mask)
        s_set = set(mask_to_list(mask))
        best_mask = mask
        best_score = s
        T = 2.0
        cooling = 0.995
        
        while time.time() < time_limit:
            if T < 0.01:
                break
            old = random.choice(list(s_set))
            new = random.randint(1, MAX_VAL)
            if new == old or new in s_set:
                T *= cooling
                continue
            old_bit = 1 << (old - 1)
            new_bit = 1 << (new - 1)
            delta = delta_score(mask, old_bit, new_bit)
            
            if delta > 0:
                mask = mask ^ old_bit ^ new_bit
                s_set.discard(old)
                s_set.add(new)
                s += delta
                if s > best_score:
                    best_score = s
                    best_mask = mask
            else:
                # Accept with probability exp(delta/T)
                if random.random() < pow(2.718281828, delta / T):
                    mask = mask ^ old_bit ^ new_bit
                    s_set.discard(old)
                    s_set.add(new)
                    s += delta
            
            T *= cooling
            # Periodically reheat if stuck
            if random.random() < 0.01:
                T = min(T * 2, 1.0)
        
        return best_mask, best_score

    def targeted_local_search(mask, max_iter=200, time_limit=None):
        s = score_mask(mask)
        s_set = set(mask_to_list(mask))
        for _ in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            elems = sorted(s_set)
            for i in range(len(elems)):
                a = elems[i]
                for j in range(i + 1, len(elems)):
                    b = elems[j]
                    d = b - a
                    c = a + 2 * d
                    if c > MAX_VAL:
                        break
                    if c not in s_set:
                        removable = [x for x in s_set if x != a and x != b]
                        if removable:
                            x_to_remove = random.choice(removable)
                            old_bit = 1 << (x_to_remove - 1)
                            new_bit = 1 << (c - 1)
                            delta = delta_score(mask, old_bit, new_bit)
                            if delta > 0:
                                mask = mask ^ old_bit ^ new_bit
                                s_set.discard(x_to_remove)
                                s_set.add(c)
                                s += delta
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                for old in list(s_set):
                    old_bit = 1 << (old - 1)
                    for _ in range(10):
                        new = random.randint(1, MAX_VAL)
                        if new == old or new in s_set:
                            continue
                        new_bit = 1 << (new - 1)
                        delta = delta_score(mask, old_bit, new_bit)
                        if delta > 0:
                            mask = mask ^ old_bit ^ new_bit
                            s_set.discard(old)
                            s_set.add(new)
                            s += delta
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        return mask, s

    # Generate initial candidates
    candidates = gen_candidates()
    best_mask = None
    best_score = 0
    
    for cand in candidates:
        m = list_to_mask(cand)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_mask = m
    
    print(f"Initial best: {best_score}", file=sys.stderr)

    # Multi-start with simulated annealing and local search
    num_starts = 6
    for start_i in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        if start_i == 0:
            current_mask = best_mask
            current_score = best_score
        else:
            # Perturb best
            current_mask = best_mask
            s_set = set(mask_to_list(current_mask))
            num_swaps = random.randint(3, 8)
            for _ in range(num_swaps):
                old = random.choice(list(s_set))
                new = random.randint(1, MAX_VAL)
                if new not in s_set:
                    old_bit = 1 << (old - 1)
                    new_bit = 1 << (new - 1)
                    current_mask = current_mask ^ old_bit ^ new_bit
                    s_set.discard(old)
                    s_set.add(new)
            current_score = score_mask(current_mask)
        
        # Phase 1: Simulated annealing
        if time.time() < deadline - 15:
            current_mask, current_score = simulated_annealing(current_mask, deadline - 15)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                print(f"SA improved: {best_score} at start {start_i}", file=sys.stderr)
        
        # Phase 2: Targeted local search
        if time.time() < deadline - 10:
            current_mask, current_score = targeted_local_search(current_mask, max_iter=200, time_limit=deadline - 10)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                print(f"Targeted improved: {best_score} at start {start_i}", file=sys.stderr)
        
        # Phase 3: Standard local search
        if time.time() < deadline - 5:
            current_mask, current_score = local_search(current_mask, max_iter=300, time_limit=deadline - 5)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                print(f"Local improved: {best_score} at start {start_i}", file=sys.stderr)
    
    # Final polish
    if time.time() < deadline - 3:
        best_mask, best_score = local_search(best_mask, max_iter=1000, time_limit=deadline - 2)
        print(f"Final best: {best_score}", file=sys.stderr)

    result = mask_to_list(best_mask)
    assert len(result) == N, f"Expected {N} elements, got {len(result)}"
    assert all(1 <= x <= MAX_VAL for x in result), "Values out of range"
    assert len(set(result)) == N, "Duplicate values"
    
    return result
