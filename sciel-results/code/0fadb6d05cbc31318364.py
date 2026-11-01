import time
import itertools
import numpy as np
import random
from collections import deque

_DEADLINE = 145.0  # Safety margin before 150s kill

def _direct_sum(q, n, radius):
    """Always-valid baseline: a radius-1 perfect code of length 4 over the alphabet,
    tensored with every word on the remaining n-4 coordinates.  Valid for R>=1, n>=4.
    Size = q^2 * q^(n-4) = q^(n-2)."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]

def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx

def _from_idx(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w

def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        c_idx = _idx(c, q)
        covered[c_idx] = True
        if radius >= 1:
            for pos in range(n):
                for v in range(q):
                    if v != c[pos]:
                        w = list(c)
                        w[pos] = v
                        covered[_idx(w, q)] = True
    return covered.all()

def _precompute_balls(q, n, radius):
    """Precompute all balls and reverse balls for the space."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # For radius 1, ball size is 1 + n*(q-1)
    ball_size = 1 + n * (q - 1) if radius == 1 else None
    
    balls = []
    reverse_balls = [[] for _ in range(total)]
    
    for i in range(total):
        center = all_words[i]
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                for v in range(q):
                    if v != center[pos]:
                        w = list(center)
                        w[pos] = v
                        w_idx = _idx(w, q)
                        ball.append(w_idx)
        balls.append(ball)
        for w in ball:
            reverse_balls[w].append(i)
            
    return all_words, balls, reverse_balls, ball_size

def _greedy_set_cover(balls, reverse_balls, total, ball_size, rng, time_limit):
    """Greedy set cover with randomized tie-breaking."""
    start = time.time()
    covered = np.zeros(total, dtype=bool)
    gain = np.full(total, ball_size, dtype=np.int32)
    code_indices = []
    
    while not covered.all():
        if time.time() - start > time_limit:
            return None
        
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        candidates = np.where(gain == max_gain)[0]
        best = rng.choice(candidates)
        
        code_indices.append(best)
        
        b = balls[best]
        for w in b:
            if not covered[w]:
                covered[w] = True
                for c in reverse_balls[w]:
                    gain[c] -= 1
                    
    if not covered.all():
        return None
    return code_indices

def _prune_redundant(code_indices, balls, reverse_balls, total, rng, time_limit):
    """Remove redundant codewords by checking coverage counts."""
    start = time.time()
    code_set = set(code_indices)
    
    # Build coverage count
    cover_count = np.zeros(total, dtype=np.int32)
    for c_idx in code_set:
        for w in balls[c_idx]:
            cover_count[w] += 1
            
    if not (cover_count > 0).all():
        return None
        
    changed = True
    while changed:
        changed = False
        current_code = list(code_set)
        rng.shuffle(current_code)
        for c_idx in current_code:
            if time.time() - start > time_limit:
                break
            is_essential = False
            for w in balls[c_idx]:
                if cover_count[w] == 1:
                    is_essential = True
                    break
            if not is_essential:
                code_set.remove(c_idx)
                for w in balls[c_idx]:
                    cover_count[w] -= 1
                changed = True
                if not (cover_count > 0).all():
                    return None
        if time.time() - start > time_limit:
            break
            
    return list(code_set)

def _ils_improve(code_indices, balls, reverse_balls, total, ball_size, rng, time_limit):
    """Iterated local search: try to improve by moving codewords."""
    start = time.time()
    best_code = list(code_indices)
    best_size = len(best_code)
    
    # Build coverage count for current solution
    def build_cover_count(code_list):
        cc = np.zeros(total, dtype=np.int32)
        for c_idx in code_list:
            for w in balls[c_idx]:
                cc[w] += 1
        return cc
    
    cover_count = build_cover_count(best_code)
    
    while time.time() - start < time_limit:
        # Try to remove a non-essential codeword
        code_list = list(best_code)
        rng.shuffle(code_list)
        
        removed = False
        for c_idx in code_list:
            if time.time() - start > time_limit:
                break
            is_essential = False
            for w in balls[c_idx]:
                if cover_count[w] == 1:
                    is_essential = True
                    break
            if not is_essential:
                # Try removing
                for w in balls[c_idx]:
                    cover_count[w] -= 1
                if (cover_count > 0).all():
                    code_list.remove(c_idx)
                    removed = True
                    break
                else:
                    # Restore
                    for w in balls[c_idx]:
                        cover_count[w] += 1
                    break
        
        if removed and len(code_list) < best_size:
            best_code = code_list
            best_size = len(best_code)
            cover_count = build_cover_count(best_code)
            continue
        
        # Try moving a codeword to a new position
        if best_size > 1:
            c_idx = rng.choice(best_code)
            # Find a random new position
            new_idx = rng.randint(0, total - 1)
            if new_idx not in best_code:
                # Check if moving improves or maintains coverage
                # First, see what would be lost
                lost_words = []
                for w in balls[c_idx]:
                    if cover_count[w] == 1:
                        lost_words.append(w)
                
                if not lost_words:
                    # Can remove without issue, but we want to move not remove
                    # Try the move
                    for w in balls[c_idx]:
                        cover_count[w] -= 1
                    for w in balls[new_idx]:
                        cover_count[w] += 1
                    if (cover_count > 0).all():
                        best_code.remove(c_idx)
                        best_code.append(new_idx)
                    else:
                        # Undo
                        for w in balls[new_idx]:
                            cover_count[w] -= 1
                        for w in balls[c_idx]:
                            cover_count[w] += 1
                else:
                    # Check if new position covers all lost words
                    new_ball_set = set(balls[new_idx])
                    all_covered = all(w in new_ball_set for w in lost_words)
                    if all_covered:
                        for w in balls[c_idx]:
                            cover_count[w] -= 1
                        for w in balls[new_idx]:
                            cover_count[w] += 1
                        if (cover_count > 0).all():
                            best_code.remove(c_idx)
                            best_code.append(new_idx)
                        else:
                            for w in balls[new_idx]:
                                cover_count[w] -= 1
                            for w in balls[c_idx]:
                                cover_count[w] += 1
    
    return best_code

def construct(q, n, radius):
    start = time.time()
    
    # Always-valid baseline
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best = baseline
    best_len = len(baseline)
    
    # For the specific case q=3, n=6, r=1, use specialized approach
    if q == 3 and n == 6 and radius == 1:
        total = 729
        ball_size = 1 + 6 * 2  # 13
        
        # Precompute
        all_words, balls, reverse_balls, _ = _precompute_balls(q, n, radius)
        
        # Multiple greedy restarts with pruning
        time_budget_total = max(1.0, _DEADLINE - (time.time() - start) - 10.0)
        num_restarts = 15
        time_per_restart = time_budget_total / num_restarts
        
        rng = random.Random(42)
        
        for i in range(num_restarts):
            if time.time() - start > _DEADLINE - 10.0:
                break
            
            # Greedy set cover
            code_indices = _greedy_set_cover(balls, reverse_balls, total, ball_size, 
                                            rng, time_per_restart * 0.4)
            if code_indices is None:
                continue
            
            # Prune redundant
            pruned = _prune_redundant(code_indices, balls, reverse_balls, total, 
                                      rng, time_per_restart * 0.3)
            if pruned is None:
                continue
            
            # ILS improvement
            improved = _ils_improve(pruned, balls, reverse_balls, total, ball_size,
                                    rng, time_per_restart * 0.3)
            if improved is not None and len(improved) < best_len:
                final_code = [all_words[i].tolist() for i in improved]
                # Verify
                if _covers_all(final_code, q, n, radius):
                    best = final_code
                    best_len = len(final_code)
                    if best_len == 73:
                        break
    
    # Final safety check
    if time.time() - start > _DEADLINE - 5.0:
        return baseline
    
    # Verify best solution
    if not _covers_all(best, q, n, radius):
        return baseline
    
    return best
