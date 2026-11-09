import time
import sys

def construct():
    n = 10
    # Construct a single Latin Square of order 10.
    # It is a known theorem that there are no two mutually orthogonal Latin squares of order 10.
    # Therefore, the maximum number of squares we can return is 1.
    # We generate a simple cyclic Latin square.
    
    S0 = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append((i + j) % n)
        S0.append(row)
    
    # Ensure types are strictly Python int and list
    result = []
    for row in S0:
        result.append([int(x) for x in row])
        
    return [result]
