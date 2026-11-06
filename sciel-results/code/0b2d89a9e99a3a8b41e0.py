import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    N = 16

    # Precompute all K4 edge sets
    k4s = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    k4s.append(((i,j), (i,k), (i,l), (j,k), (j,l), (k,l)))

    def is_valid(cm):
        for edges in k4s:
            c0 = cm[edges[0]]; c1 = cm[edges[1]]; c2 = cm[edges[2]]
            c3 = cm[edges[3]]; c4 = cm[edges[4]]; c5 = cm[edges[5]]
            # Check no colour appears 3+ times
            if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c1 == c4 or c0 == c1 == c5:
                return False
            if c0 == c2 == c3 or c0 == c2 == c4 or c0 == c2 == c5:
                return False
            if c0 == c3 == c4 or c0 == c3 == c5:
                return False
            if c0 == c4 == c5:
                return False
            if c1 == c2 == c3 or c1 == c2 == c4 or c1 == c2 == c5:
                return False
            if c1 == c3 == c4 or c1 == c3 == c5:
                return False
            if c1 == c4 == c5:
                return False
            if c2 == c3 == c4 or c2 == c3 == c5:
                return False
            if c2 == c4 == c5:
                return False
            if c3 == c4 == c5:
                return False
        return True

    def to_list(cm):
        result = []
        for i in range(N):
            for j in range(i+1, N):
                result.append(cm[(i,j)])
        return result

    def nc(cm):
        return len(set(cm.values()))

    # Initial valid solution
    init = [(i + j) % 17 for i in range(N) for j in range(i+1, N)]
    cm0 = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            cm0[(i,j)] = init[idx]
            idx += 1

    best = init
    best_nc = 17
    random.seed(42)

    while time.time() < deadline:
        if best_nc <= 13:
            break

        # Try to reduce from current best
        cm = dict(cm0)
        cur_nc = nc(cm)

        for _ in range(100):
            if time.time() >= deadline:
                break
            if cur_nc <= 13:
                break

            # Pick a random colour and try to eliminate it
            colours = sorted(set(cm.values()))
            if len(colours) <= 13:
                break
            c = random.choice(colours)
            edges_c = [(i,j) for i in range(N) for j in range(i+1,N) if cm[(i,j)] == c]
            if not edges_c:
                continue

            # Try reassigning edges of colour c
            saved = {e: cm[e] for e in edges_c}
            success = True
            for e in edges_c:
                found = False
                for new_c in colours:
                    if new_c == c:
                        continue
                    cm[e] = new_c
                    if is_valid(cm):
                        found = True
                        break
                if not found:
                    # Try all colours including c
                    for new_c in range(max(colours)+1):
                        cm[e] = new_c
                        if is_valid(cm):
                            found = True
                            break
                if not found:
                    for e2 in edges_c:
                        cm[e2] = saved[e2]
                    success = False
                    break

            if success:
                new_nc = nc(cm)
                if new_nc < cur_nc:
                    cur_nc = new_nc
                    if cur_nc < best_nc:
                        best_nc = cur_nc
                        best = to_list(cm)

        # Random restart
        for _ in range(30):
            if time.time() >= deadline:
                break
            if best_nc <= 13:
                break
            cm2 = {}
            for i in range(N):
                for j in range(i+1, N):
                    cm2[(i,j)] = random.randint(0, best_nc - 1)
            if is_valid(cm2):
                n2 = nc(cm2)
                if n2 < best_nc:
                    best_nc = n2
                    best = to_list(cm2)

    # Remap to 0..119
    if best:
        mc = max(best)
        if mc > 119:
            cm = {}
            idx = 0
            for i in range(N):
                for j in range(i+1, N):
                    cm[(i,j)] = best[idx]
                    idx += 1
            uniq = sorted(set(cm.values()))
            remap = {c: i for i, c in enumerate(uniq)}
            for k in cm:
                cm[k] = remap[cm[k]]
            best = to_list(cm)

    return best
