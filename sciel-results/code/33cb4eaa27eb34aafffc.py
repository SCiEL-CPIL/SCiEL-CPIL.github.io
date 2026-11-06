# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Initial Greedy (Increasing order)
    S = set()
    for x in range(1, 257):
        if not _is_conflict(S, x):
            S.add(x)
            
    def get_conflict_set(S, b):
        conflicts = set()
        for u in S:
            # 1. b is middle: u + v = 2b => v = 2b - u
            v = 2 * b - u
            if v in S and v != u:
                conflicts.add(u)
                if len(conflicts) > 1:
                    return conflicts
            # 2. u is middle: b + v = 2u => v = 2u - b
            v = 2 * u - b
            if v in S and v != u:
                conflicts.add(u)
                if len(conflicts) > 1:
                    return conflicts
            # 3. v is middle: b + u = 2v => v = (b+u)/2
            if (b + u) % 2 == 0:
                v = (b + u) // 2
                if v in S and v != u:
                    conflicts.add(u)
                    if len(conflicts) > 1:
                        return conflicts
        return conflicts

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        candidates = []
        for b in range(1, 257):
            if b in S:
                continue
            C_b = get_conflict_set(S, b)
            if len(C_b) == 1:
                a = next(iter(C_b))
                # Swap a with b if it improves score (1/b > 1/a => a > b)
                if a > b:
                    candidates.append((a, b))
        
        if candidates:
            # Sort by gain: 1/b - 1/a
            candidates.sort(key=lambda item: 1.0/item[1] - 1.0/item[0], reverse=True)
            a, b = candidates[0]
            S.remove(a)
            S.add(b)
            improved = True
            
    return list(S)

def _is_conflict(S, x):
    for u in S:
        v = 2 * x - u
        if v in S and v != u:
            return True
        v = 2 * u - x
        if v in S and v != u:
            return True
        if (x + u) % 2 == 0:
            v = (x + u) // 2
            if v in S and v != u:
                return True
    return False
# EVOLVE-BLOCK-END
