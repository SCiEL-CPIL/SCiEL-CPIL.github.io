import time
import random
import numpy as np

N = 512
TIME_LIMIT = 55.0  # safety margin under 90s kill

# Precompute all 4-APs and their incidence structure
def precompute():
    ap_by_elem = [[] for _ in range(N)]  # 0-indexed: ap_by_elem[x] = list of (o1, o2, o3)
    aps = []
    for a in range(N):
        for d in range(1, N // 3 + 1):
            a3 = a + 3 * d
            if a3 >= N:
                break
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            ap_id = len(aps)
            aps.append((p0, p1, p2, p3))
            ap_by_elem[p0].append((p1, p2, p3))
            ap_by_elem[p1].append((p0, p2, p3))
            ap_by_elem[p2].append((p0, p1, p3))
            ap_by_elem[p3].append((p0, p1, p2))
    return aps, ap_by_elem

APS, AP_BY_ELEM = precompute()
NUM_APS = len(APS)

def get_base3_set(n):
    """Salem-Spencer type set: numbers with only 0,1 in base 3."""
    s = set()
    for x in range(1, n + 1):
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            s.add(x)
    return s

def mask_from_set(S):
    """Convert 1-indexed set to 0-indexed bitmask as numpy bool array."""
    mask = np.zeros(N, dtype=bool)
    for x in S:
        mask[x - 1] = True
    return mask

def set_from_mask(mask):
    """Convert 0-indexed bool array to 1-indexed sorted list."""
    return [i + 1 for i in range(N) if mask[i]]

def can_add_mask(mask, x):
    """Check if adding element x (0-indexed) to set represented by mask creates a 4-AP."""
    for (o1, o2, o3) in AP_BY_ELEM[x]:
        if mask[o1] and mask[o2] and mask[o3]:
            return False
    return True

def greedy_fill_mask(mask, candidates_order):
    """Greedy fill: try to add elements in given order."""
    for x in candidates_order:
        if not mask[x] and can_add_mask(mask, x):
            mask[x] = True
    return mask

def local_search_mask(mask, deadline):
    """
    Local search: repeatedly try to add safe elements; if stuck, remove
    the element in the most 'blocking' APs (APs where other 3 are in set).
    """
    current_mask = mask.copy()
    current_size = current_mask.sum()
    best_mask = current_mask.copy()
    best_size = current_size

    # Precompute for each element, the list of AP indices it's in
    # For speed, we'll track AP counts incrementally
    # ap_count[i] = number of elements from AP i that are in the set
    ap_count = np.zeros(NUM_APS, dtype=np.int32)
    for i in range(N):
        if current_mask[i]:
            for (o1, o2, o3) in AP_BY_ELEM[i]:
                # Find APs containing i
                pass  # We'll recompute or use a different approach

    # Actually, let's use a simpler approach: maintain ap_count properly
    # For each AP, count how many of its 4 elements are in the set
    for ap_idx, (p0, p1, p2, p3) in enumerate(APS):
        cnt = 0
        if current_mask[p0]: cnt += 1
        if current_mask[p1]: cnt += 1
        if current_mask[p2]: cnt += 1
        if current_mask[p3]: cnt += 1
        ap_count[ap_idx] = cnt

    # For each element, list of AP indices it belongs to
    elem_ap_indices = [[] for _ in range(N)]
    for ap_idx, (p0, p1, p2, p3) in enumerate(APS):
        elem_ap_indices[p0].append(ap_idx)
        elem_ap_indices[p1].append(ap_idx)
        elem_ap_indices[p2].append(ap_idx)
        elem_ap_indices[p3].append(ap_idx)

    def can_add_fast(x):
        """Check if x can be added using ap_count."""
        for ap_idx in elem_ap_indices[x]:
            if ap_count[ap_idx] >= 3:
                return False
        return True

    def do_add(x):
        current_mask[x] = True
        for ap_idx in elem_ap_indices[x]:
            ap_count[ap_idx] += 1

    def do_remove(x):
        current_mask[x] = False
        for ap_idx in elem_ap_indices[x]:
            ap_count[ap_idx] -= 1

    for iteration in range(500000):
        if time.time() > deadline:
            break

        # Try to add any safe element
        added = False
        # Scan in random order for diversity
        for x in range(N):
            if not current_mask[x] and can_add_fast(x):
                do_add(x)
                added = True
                break

        if added:
            if current_mask.sum() > best_size:
                best_size = current_mask.sum()
                best_mask = current_mask.copy()
            continue

        # No safe element: find element with most blocking APs
        best_e = None
        best_block = 0
        for e in range(N):
            if not current_mask[e]:
                continue
            block = 0
            for ap_idx in elem_ap_indices[e]:
                if ap_count[ap_idx] == 4:
                    block += 1
            if block > best_block:
                best_block = block
                best_e = e

        if best_e is None or best_block == 0:
            # No blocking APs, meaning no AP has all 4 in set.
            # But we couldn't add any element, so every candidate x has some AP with 3 in set.
            # This means there are APs with exactly 3 in set.
            # Let's try removing from APs with count 3
            best_e = None
            best_freed = 0
            for e in range(N):
                if not current_mask[e]:
                    continue
                freed = 0
                for ap_idx in elem_ap_indices[e]:
                    if ap_count[ap_idx] == 3:
                        freed += 1
                if freed > best_freed:
                    best_freed = freed
                    best_e = e

            if best_e is None or best_freed == 0:
                break

        do_remove(best_e)

    return best_mask, best_size

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Start with base-3 set
    base_S = get_base3_set(N)
    mask = mask_from_set(base_S)

    best_mask = mask.copy()
    best_size = mask.sum()

    # Greedy fill on base
    candidates = [x for x in range(N) if not mask[x]]
    random.shuffle(candidates)
    mask = greedy_fill_mask(mask, candidates)
    if mask.sum() > best_size:
        best_size = mask.sum()
        best_mask = mask.copy()

    # Local search
    while time.time() < deadline - 5:
        # Perturb: remove some random elements
        current_mask = best_mask.copy()
        removal_rate = random.choice([0.05, 0.1, 0.15, 0.2])
        num_remove = int(len(set_from_mask(current_mask)) * removal_rate)
        if num_remove > 0:
            in_set = [x for x in range(N) if current_mask[x]]
            to_remove = random.sample(in_set, min(num_remove, len(in_set)))
            for x in to_remove:
                current_mask[x] = False

        # Greedy fill
        candidates = [x for x in range(N) if not current_mask[x]]
        random.shuffle(candidates)
        current_mask = greedy_fill_mask(current_mask, candidates)

        # Local search
        current_mask, current_size = local_search_mask(current_mask, deadline - 2)

        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask.copy()

    return set_from_mask(best_mask)
