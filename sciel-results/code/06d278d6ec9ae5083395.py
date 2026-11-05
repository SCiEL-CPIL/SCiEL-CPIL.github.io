import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = 128
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    index_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(i)
            
    def evaluate(colors):
        bad = 0
        for ap in aps:
            if len({colors[i] for i in ap}) < 3:
                bad += 1
        return bad

    def local_search(k, time_limit):
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_bad = evaluate(colors)
        if current_bad == 0:
            return colors
            
        best_colors = colors[:]
        best_bad = current_bad
        
        t = time.time()
        while time.time() - t < time_limit:
            i = random.randint(0, n - 1)
            old_color = colors[i]
            affected = index_to_aps[i]
            
            best_new_color = old_color
            min_bad = current_bad
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                colors[i] = new_color
                # Calculate bad count for affected APs
                # We need to know how many of these were bad before
                # This requires tracking state.
                # Simpler: Just re-evaluate the whole board? No, too slow if we do it for every color.
                # Let's just evaluate the affected APs and assume the rest stay same.
                # But we don't know if they were bad before.
                
                # Alternative: Just use global evaluate for the best candidate?
                # No, let's stick to a simpler heuristic:
                # Pick random i, random new_color.
                # If evaluate(colors) < current_bad, accept.
                # This is slow.
                
                # Let's optimize: 
                # Precompute for each AP its current distinct count?
                # Too complex.
                
                # Given n=128, global evaluate is ~2700 iterations.
                # If we do 1000 steps, that's 2.7M iterations. Fine.
                # Let's just do random steps with global eval.
                pass
            
            colors[i] = old_color
            
            # Random step
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue
            colors[i] = new_color
            new_bad = evaluate(colors)
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
                if current_bad == 0:
                    return colors
            else:
                colors[i] = old_color
                
        return best_colors

    # Try k=3
    if time.time() < deadline:
        res = local_search(3, 40)
        if evaluate(res) == 0:
            return res
            
    # Try k=4
    if time.time() < deadline:
        res = local_search(4, 30)
        if evaluate(res) == 0:
            return res

    # Try k=5
    if time.time() < deadline:
        res = local_search(5, 10)
        if evaluate(res) == 0:
            return res

    return res
