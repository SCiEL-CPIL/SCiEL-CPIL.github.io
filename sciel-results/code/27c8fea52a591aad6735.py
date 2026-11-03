from itertools import product

def get_rank(ns, ms, ps):
    if ns == 2 and ms == 2 and ps == 2: return 7
    if ns == 2 and ms == 2 and ps == 1: return 4
    if ns == 1 and ms == 2 and ps == 2: return 4
    if ns == 1 and ms == 2 and ps == 1: return 2
    if ns == 2 and ms == 1 and ps == 2: return 4
    if ns == 2 and ms == 1 and ps == 1: return 2
    if ns == 1 and ms == 1 and ps == 2: return 2
    if ns == 1 and ms == 1 and ps == 1: return 1
    if ns == 2 and ms == 2 and ps == 3: return 11
    if ns == 1 and ms == 2 and ps == 3: return 6
    if ns == 2 and ms == 1 and ps == 3: return 6
    if ns == 1 and ms == 1 and ps == 3: return 3
    if ns == 2 and ms == 3 and ps == 2: return 12  # 2 * (1x3x2 rank 6)
    if ns == 3 and ms == 2 and ps == 2: return 12  # 3 * (1x2x2 rank 4)
    if ns == 3 and ms == 2 and ps == 1: return 6
    if ns == 2 and ms == 3 and ps == 1: return 6
    if ns == 1 and ms == 3 and ps == 2: return 6
    if ns == 3 and ms == 1 and ps == 2: return 6
    if ns == 2 and ms == 1 and ps == 3: return 6
    if ns == 1 and ms == 3 and ps == 1: return 3
    if ns == 3 and ms == 1 and ps == 1: return 3
    return ns * ms * ps

def get_splits(total, max_parts=5):
    """Generate all compositions of total into at most max_parts positive integers."""
    if total == 1:
        yield [1]
        return
    for first in range(1, total):
        for rest in get_splits(total - first, max_parts - 1):
            yield [first] + rest

best_rank = float('inf')
best_split = None

for n_split in get_splits(3, 3):
    for m_split in get_splits(4, 4):
        for p_split in get_splits(7, 5):
            rank = 0
            for ns in n_split:
                for ms in m_split:
                    for ps in p_split:
                        rank += get_rank(ns, ms, ps)
            if rank < best_rank:
                best_rank = rank
                best_split = (n_split, m_split, p_split)

print(f"Best rank: {best_rank}, split: {best_split}")
