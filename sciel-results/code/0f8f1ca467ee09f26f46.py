import time

# EVOLVE-BLOCK-START

def construct():
    n = 10
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin
    
    # Generate Square 1
    # S1[i][j] = 5 * ((i//5 + j//5) % 2) + ((i%5 + j%5) % 5)
    sq1 = []
    for i in range(n):
        row = []
        for j in range(n):
            u = i // 5
            x = i % 5
            v = j // 5
            y = j % 5
            
            val1 = 5 * ((u + v) % 2) + ((x + y) % 5)
            row.append(int(val1))
        sq1.append(row)
        
    # Generate Square 2
    # S2[i][j] = 5 * (j//5 % 2) + ((i%5 + 2*(j%5)) % 5)
    sq2 = []
    for i in range(n):
        row = []
        for j in range(n):
            u = i // 5
            x = i % 5
            v = j // 5
            y = j % 5
            
            val2 = 5 * (v % 2) + ((x + 2 * y) % 5)
            row.append(int(val2))
        sq2.append(row)
        
    # Verify types and orthogonality just in case (fast check)
    # Check Latin property
    for sq in [sq1, sq2]:
        for r in range(n):
            if sorted(sq[r]) != list(range(n)):
                return [sq1] # Fallback to single square if broken
        for c in range(n):
            col = [sq[r][c] for r in range(n)]
            if sorted(col) != list(range(n)):
                return [sq1]
                
    # Check Orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return [sq1]
            pairs.add(pair)
            
    return [sq1, sq2]

# EVOLVE-BLOCK-END
