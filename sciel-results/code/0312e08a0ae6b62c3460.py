import time
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    # Convert to int/float
    result = {}
    for key in ['U', 'V', 'W']:
        mat = result[key] = []
        for row in {'U': U, 'V': V, 'W': W}[key]:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(int(val))
                else:
                    new_row.append(float(val))
            mat.append(new_row)
    
    return result
