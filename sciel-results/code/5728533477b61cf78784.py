import time

def construct():
    answers = []
    start_time = time.time()
    # Safety margin: 55s
    deadline = start_time + 55.0
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            if y_min <= x:
                y_min = x + 1
                
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                    
                D = A * y - B
                if D <= 0:
                    continue
                    
                if (B * y) % D == 0:
                    z = (B * y) // D
                    if z > y and z <= 1000000000000:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
