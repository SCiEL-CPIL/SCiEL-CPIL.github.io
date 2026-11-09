import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 4, 4, 6
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    terms = [
        {
            "A": [((0,0), 1), ((1,1), 1)],
            "B": [((0,0), 1), ((1,2), 1)],
            "C": [((0,0), 1), ((1,1), 1)]
        },
        {
            "A": [((1,0), 1), ((1,1), 1)],
            "B": [((0,0), 1)],
            "C": [((1,0), 1)]
        },
        {
            "A": [((0,0), 1)],
            "B": [((0,1), 1), ((1,2), -1)],
            "C": [((0,1), 1)]
        },
        {
            "A": [((1,1), 1)],
            "B": [((1,0), 1), ((0,0), -1)],
            "C": [((1,0), 1)]
        },
        {
            "A": [((0,0), 1), ((0,1), 1)],
            "B": [((1,2), 1)],
            "C": [((0,0), -1), ((0,1), 1)]
        },
        {
            "A": [((1,0), 1), ((0,0), -1)],
            "B": [((0,0), 1), ((0,1), 1)],
            "C": [((1,1), 1)]
        },
        {
            "A": [((0,1), 1), ((1,1), -1)],
            "B": [((1,0), 1), ((1,2), 1)],
            "C": [((0,0), 1)]
        }
    ]
    
    for t in terms:
        for u in range(2):
            for v in range(2):
                for w in range(3):
                    for (ar, ac), sign in t["A"]:
                        i = 2 * ar + u
                        j = 2 * ac + v
                        U[i * m + j][col] = sign
                    
                    for (br, bc), sign in t["B"]:
                        j = 2 * br + v
                        k = 3 * bc + w
                        V[j * p + k][col] = sign
                    
                    for (cr, cc), sign in t["C"]:
                        i = 2 * cr + u
                        k = 3 * cc + w
                        W[k * n + i][col] = sign
                    
                    col += 1
                    
                    if time.time() > deadline - 5.0:
                        break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
            
    return {"U": U, "V": V, "W": W}
