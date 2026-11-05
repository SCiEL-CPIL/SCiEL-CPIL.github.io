"""Improved snake-in-the-box for Q_9 using Simulated Annealing."""
import random
import sys
import time
import math

N = 9
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using Simulated Annealing.
    """
    rng = random.Random(12345)
    
    def can_add_head(v, path, used):
        if v in used:
            return False
        if not path:
            return True
        if (v ^ path[0]).bit_count() != 1:
            return False
        for i in range(1, len(path)):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True

    def can_add_tail(v, path, used):
        if v in used:
            return False
        if not path:
            return True
        if (v ^ path[-1]).bit_count() != 1:
            return False
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True

    def generate_initial_snake():
        start = rng.randint(0, (1 << N) - 1)
        path = [start]
        used = {start}
        for _ in range(50):
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            if not candidates:
                break
            v = rng.choice(candidates)
            path.append(v)
            used.add(v)
        return path, used

    current_path, current_used = generate_initial_snake()
    best_path = current_path[:]
    best_len = len(current_path)
    
    T = 1.0
    cooling_rate = 0.999
    T_min = 0.05
    
    start_time = time.time()
    time_limit = 115
    
    while time.time() - start_time < time_limit:
        # 60% extend, 40% contract
        if rng.random() < 0.6:
            # Try to extend
            if rng.random() < 0.5:
                # Head
                cur = current_path[0]
                candidates = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if can_add_head(v, current_path, current_used):
                        candidates.append(v)
                if candidates:
                    v = rng.choice(candidates)
                    # Always accept improvement
                    current_path.insert(0, v)
                    current_used.add(v)
                    if len(current_path) > best_len:
                        best_len = len(current_path)
                        best_path = current_path[:]
            else:
                # Tail
                cur = current_path[-1]
                candidates = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if can_add_tail(v, current_path, current_used):
                        candidates.append(v)
                if candidates:
                    v = rng.choice(candidates)
                    current_path.append(v)
                    current_used.add(v)
                    if len(current_path) > best_len:
                        best_len = len(current_path)
                        best_path = current_path[:]
        else:
            # Try to contract
            if len(current_path) > 1:
                # Simulate contraction
                if rng.random() < 0.5:
                    v = current_path[0]
                    # Check acceptance
                    if rng.random() < math.exp(-1.0 / T):
                        current_path.pop(0)
                        current_used.discard(v)
                else:
                    v = current_path[-1]
                    if rng.random() < math.exp(-1.0 / T):
                        current_path.pop()
                        current_used.discard(v)
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_path
# EVOLVE-BLOCK-END
