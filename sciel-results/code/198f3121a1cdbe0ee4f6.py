import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Use k=6 active vertices for dense comparability
    k = 6
    total_masks = 1 << k  # 64 subsets
    
    # Precompute comparability for all pairs of masks
    # comp[i][j] = True if mask i and mask j are comparable (one is subset of other)
    # We'll use bit operations for speed
    
    def is_comparable(a, b):
        """Check if two masks are comparable (one is strict subset of other or vice versa)"""
        # a subset of b: (a & ~b) == 0
        # b subset of a: (b & ~a) == 0
        # They are comparable if either holds (and they're not equal, but we handle distinct)
        return (a & ~b) == 0 or (b & ~a) == 0
    
    # Precompute comparability matrix
    comp = [[False] * total_masks for _ in range(total_masks)]
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if is_comparable(i, j):
                comp[i][j] = True
                comp[j][i] = True
    
    # Initial family: sort by (size, mask) and take first m
    # This gives us all small sets first, which are subsets of many others
    masks_by_size = sorted(range(total_masks), key=lambda x: (bin(x).count('1'), x))
    family = set(masks_by_size[:m])
    
    # Calculate score of a family
    def calc_score(fam):
        score = 0
        fam_list = list(fam)
        for i in range(len(fam_list)):
            for j in range(i + 1, len(fam_list)):
                if comp[fam_list[i]][fam_list[j]]:
                    score += 1
        return score
    
    current_score = calc_score(family)
    
    # Hill climbing: try swapping elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in list(family):
            for j in range(total_masks):
                if j not in family:
                    # Try swapping i with j
                    new_family = family - {i} | {j}
                    new_score = calc_score(new_family)
                    if new_score > current_score:
                        family = new_family
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    # Convert masks to lists of vertices
    result = []
    for mask in family:
        verts = [b for b in range(k) if (mask >> b) & 1]
        result.append(verts)
    
    # Ensure exactly m distinct lists
    # They should be distinct since masks are distinct
    assert len(result) == m
    assert len(set(tuple(x) for x in result)) == m
    
    return result
