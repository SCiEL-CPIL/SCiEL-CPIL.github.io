def construct():
    import time
    import random
    start_time = time.time()
    deadline = start_time + 115

    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def paley_coloring(n):
        if n < 2:
            return []
        if not is_prime(n) or n % 4 != 1:
            return None
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    def colors_to_adj(n, colors):
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
        return adj0, adj1

    def count_k222(adj, n):
        count = 0
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                c_uv = au & adj[v]
                c_uv &= ~(1 << u)
                c_uv &= ~(1 << v)
                # Iterate x in c_uv
                # We can extract bits
                # But n is small, loop is fine
                for x in range(n):
                    if not (c_uv & (1 << x)):
                        continue
                    ax = adj[x]
                    for y in range(x + 1, n):
                        if not (c_uv & (1 << y)):
                            continue
                        ay = adj[y]
                        common = ax & ay & c_uv
                        if common.bit_count() >= 2:
                            count += 1
        return count

    def objective(adj0, adj1, n):
        return count_k222(adj0, n) + count_k222(adj1, n)

    # 1. Try n=29 Paley
    colors29 = paley_coloring(29)
    if colors29:
        adj0_29, adj1_29 = colors_to_adj(29, colors29)
        if objective(adj0_29, adj1_29, 29) == 0:
            best_result = {'n': 29, 'colors': colors29}
        else:
            best_result = {'n': 1, 'colors': []}
    else:
        best_result = {'n': 1, 'colors': []}

    # 2. Try to find n=30
    if time.time() < deadline:
        n = 30
        # Initialize from Paley 29 + random vertex 29
        # Extend adj0_29, adj1_29 to n=30
        adj0 = [0] * n
        adj1 = [0] * n
        for i in range(29):
            adj0[i] = adj0_29[i]
            adj1[i] = adj1_29[i]
        
        # Connect vertex 29
        # Try to minimize immediate violations
        # Just random for now, local search will fix
        for i in range(29):
            if random.random() < 0.5:
                adj0[i] |= (1 << 29)
                adj0[29] |= (1 << i)
            else:
                adj1[i] |= (1 << 29)
                adj1[29] |= (1 << i)
        
        current_obj = objective(adj0, adj1, n)
        
        # Local Search
        steps = 0
        while time.time() < deadline - 5:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Flip edge (i, j)
            bit = 1 << j
            bit_i = 1 << i
            
            # Determine current color
            # If bit is set in adj0[i], it's color 0
            is_color0 = bool(adj0[i] & bit)
            
            # Calculate delta
            # This is tricky because count_k222 is global.
            # Recalculating full objective is O(n^4) ~ 10k ops.
            # 10k ops * 1000 steps = 10M ops. Should be fine in 110s?
            # Python is slow. 10M simple ops might take 1-5s.
            # So we can do maybe 5000-10000 steps.
            
            # Flip
            if is_color0:
                adj0[i] ^= bit
                adj0[j] ^= bit_i
                adj1[i] |= bit
                adj1[j] |= bit_i
            else:
                adj1[i] ^= bit
                adj1[j] ^= bit_i
                adj0[i] |= bit
                adj0[j] |= bit_i
            
            new_obj = objective(adj0, adj1, n)
            
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj == 0:
                    # Found valid coloring
                    # Convert back to colors list
                    colors = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj0[u] & (1 << v):
                                colors.append(0)
                            else:
                                colors.append(1)
                    return {'n': n, 'colors': colors}
            else:
                # Revert
                if is_color0:
                    adj0[i] ^= bit
                    adj0[j] ^= bit_i
                    adj1[i] ^= bit
                    adj1[j] ^= bit_i
                else:
                    adj1[i] ^= bit
                    adj1[j] ^= bit_i
                    adj0[i] ^= bit
                    adj0[j] ^= bit_i
            
            steps += 1
            if steps % 1000 == 0 and time.time() > deadline - 10:
                break

    return best_result
