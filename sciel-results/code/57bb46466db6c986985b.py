# Inside SA loop
if random.random() < 0.1:
    # Double swap
    x1, x2 = random.sample(S_list, 2)
    while True:
        y1 = random.randint(1, 512)
        if y1 not in S_set:
            break
    while True:
        y2 = random.randint(1, 512)
        if y2 not in S_set and y2 != y1:
            break
            
    new_mask = mask ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
    new_score = _score_mask(new_mask)
    
    delta = new_score - cur_score
    if delta > 0 or random.random() < exp(delta / T):
        S_set.discard(x1)
        S_set.discard(x2)
        S_set.add(y1)
        S_set.add(y2)
        S_list.remove(x1)
        S_list.remove(x2)
        S_list.append(y1)
        S_list.append(y2)
        mask = new_mask
        cur_score = new_score
        no_improve = 0
        if cur_score > best_score:
            best_score = cur_score
            best_mask = mask
    else:
        no_improve += 1
else:
    # Single swap (as before)
    ...
