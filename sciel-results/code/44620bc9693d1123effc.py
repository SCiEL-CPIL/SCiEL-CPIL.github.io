import time
import math
import random
from itertools import combinations

def _score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 0.0
    diff_set = set()
    sum_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
            sum_set.add(a + b)
    num_diff = len(diff_set)
    num_sum = len(sum_set)
    if num_sum == 0:
        return 0.0
    ratio = num_diff / num_sum
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_set = [0, 1, 3]
    best_score = _score_set(best_set)

    # Phase 1: Exhaustive search over subsets of {0,...,18}
    for max_elem in range(1, 19):
        if time.time() > deadline:
            break
        inner = list(range(1, max_elem))
        max_size = min(10, max_elem)
        for size in range(2, max_size + 1):
            if time.time() > deadline:
                break
            need = size - 2
            if need < 0 or need > len(inner):
                continue
            for subset in combinations(inner, need):
                if time.time() > deadline:
                    break
                candidate = sorted([0] + list(subset) + [max_elem])
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_set = list(candidate)

    # Phase 2: Simulated annealing
    if time.time() < deadline:
        current_set = sorted(set(best_set))
        current_score = best_score
        no_improve = 0
        temp = 0.03
        max_iters = 100000

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > 30000:
                break

            move_type = random.random()

            if move_type < 0.3:
                upper = max(8, 2 * max(current_set) + 8)
                new_elem = random.randint(1, upper)
                if new_elem not in current_set:
                    candidate = sorted(current_set + [new_elem])
                    cand_score = _score_set(candidate)
                    delta = cand_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
                        current_set = candidate
                        current_score = cand_score
                        if cand_score > best_score:
                            best_score = cand_score
                            best_set = list(current_set)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
            elif move_type < 0.55:
                if len(current_set) > 2:
                    removable = [x for x in current_set if x != 0]
                    if removable:
                        rem = random.choice(removable)
                        candidate = sorted([x for x in current_set if x != rem])
                        cand_score = _score_set(candidate)
                        delta = cand_score - current_score
                        if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
                            current_set = candidate
                            current_score = cand_score
                            if cand_score > best_score:
                                best_score = cand_score
                                best_set = list(current_set)
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
            elif move_type < 0.8:
                if len(current_set) > 2:
                    replaceable = [x for x in current_set if x != 0]
                    if replaceable:
                        old = random.choice(replaceable)
                        upper = max(8, 2 * max(current_set) + 8)
                        new = random.randint(1, upper)
                        if new not in current_set:
                            candidate = sorted([x if x != old else new for x in current_set])
                            cand_score = _score_set(candidate)
                            delta = cand_score - current_score
                            if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
                                current_set = candidate
                                current_score = cand_score
                                if cand_score > best_score:
                                    best_score = cand_score
                                    best_set = list(current_set)
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                no_improve += 1
            else:
                if len(current_set) > 1:
                    changeable = [x for x in current_set if x != 0]
                    if changeable:
                        old = random.choice(changeable)
                        shift = random.choice([-3, -2, -1, 1, 2, 3, 4])
                        new = old + shift
                        if new > 0 and new not in current_set:
                            candidate = sorted([x if x != old else new for x in current_set])
                            cand_score = _score_set(candidate)
                            delta = cand_score - current_score
                            if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
                                current_set = candidate
                                current_score = cand_score
                                if cand_score > best_score:
                                    best_score = cand_score
                                    best_set = list(current_set)
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                no_improve += 1

            temp *= 0.9998

    # Phase 3: Greedy polish
    if time.time() < deadline:
        current = sorted(set(best_set))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for elem in list(current):
                if elem == 0:
                    continue
                candidate = [x for x in current if x != elem]
                if len(candidate) < 2:
                    continue
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    current = candidate
                    improved = True
                    break
            if improved:
                continue
            max_c = max(current)
            for new_elem in range(1, max_c + 15):
                if new_elem in current:
                    continue
                candidate = sorted(current + [new_elem])
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    current = candidate
                    improved = True
                    break

    return sorted(best_set)
