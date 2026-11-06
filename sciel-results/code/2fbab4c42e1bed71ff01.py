def is_valid_candidate(v, path, used):
    if v in used:
        return False
    for i in range(len(path) - 1):
        p = path[i]
        if (v ^ p).bit_count() == 1:
            return False
    return True

def get_lookahead_score(v, cur, path, used):
    score = 0
    for b in range(N):
        u = v ^ (1 << b)
        if u == cur:
            continue
        if u in used:
            continue
        valid = True
        for i in range(len(path) - 1):
            p = path[i]
            if (u ^ p).bit_count() == 1:
                valid = False
                break
        if valid:
            score += 1
    return score
