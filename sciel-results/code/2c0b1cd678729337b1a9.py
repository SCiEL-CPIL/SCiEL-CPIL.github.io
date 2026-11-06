def construct():
    from collections import defaultdict
    
    # Enumerate all forbidden triples (a,b,c) distinct, in [1,180], 1/a=1/b+1/c
    triples = []
    for b in range(1, 181):
        for c in range(b+1, 181):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a <= 180 and a != b and a != c:
                    triples.append(tuple(sorted((a, b, c))))
    
    # Remove duplicates
    triples = list(set(triples))
    
    def greedy_remove(triple_list, start_order=None):
        remaining = set(range(1, 181))
        if start_order:
            # Try to keep elements in start_order preference
            pass
        # Greedy: repeatedly remove element in most active triples
        while True:
            active = [t for t in triple_list if all(e in remaining for e in t)]
            if not active:
                break
            cnt = defaultdict(int)
            for t in active:
                for e in t:
                    cnt[e] += 1
            worst = max(cnt.keys(), key=lambda e: (cnt[e], -e))
            remaining.remove(worst)
        return len(remaining), remaining
    
    # Try default greedy
    size1, set1 = greedy_remove(triples)
    
    # Try greedy preferring to keep larger numbers
    def greedy_keep_large(triple_list):
        remaining = set(range(1, 181))
        while True:
            active = [t for t in triple_list if all(e in remaining for e in t)]
            if not active:
                break
            cnt = defaultdict(int)
            for t in active:
                for e in t:
                    cnt[e] += 1
            # Among elements with max count, remove the smallest
            worst = min(cnt.keys(), key=lambda e: (-cnt[e], e))
            remaining.remove(worst)
        return len(remaining), remaining
    
    size2, set2 = greedy_keep_large(triples)
    
    if size2 >= size1:
        return list(sorted(set2))
    else:
        return list(sorted(set1))
