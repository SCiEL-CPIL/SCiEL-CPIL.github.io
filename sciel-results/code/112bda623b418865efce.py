import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds out of 120s limit? No, limit is 120s. Use 110s.
    
    n = 29
    # Generate squares mod 29
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
                
    return {'n': n, 'colors': colors}
