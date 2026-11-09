import numpy as np
import math
import random

N = 11

def construct():
    # Initialize with a hexagonal grid pattern
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        inner.append([d * math.cos(angle), d * math.sin(angle), 0.0])
    for i in range(4):
        angle = i * math.pi / 2
        r = d * 2
        inner.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        
    while len(inner) < N:
        inner.append([random.uniform(-5, 5), random.uniform(-5, 5), 0.0])
    if len(inner) > N:
        inner = inner[:N]

    def get_S(inner):
        max_proj = 0
        for i in range(len(inner)):
            cx, cy, ang = inner[i]
            ang_rad = math.radians(ang)
            for j in range(6):
                v_ang = ang_rad + j * math.pi / 3
                vx = cx + math.cos(v_ang)
                vy = cy + math.sin(v_ang)
                for k in range(6):
                    n_ang = math.radians(30 + k * 60)
                    proj = vx * math.cos(n_ang) + vy * math.sin(n_ang)
                    if proj > max_proj:
                        max_proj = proj
        return max_proj * 2 / math.sqrt(3)

    def resolve_overlaps(inner, iterations=20):
        for _ in range(iterations):
            moved = False
            for i in range(len(inner)):
                for j in range(i+1, len(inner)):
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    d = math.sqrt(dx*dx + dy*dy)
                    min_d = 1.73
                    if d < min_d:
                        if d < 1e-6:
                            dx = random.uniform(-0.1, 0.1)
                            dy = random.uniform(-0.1, 0.1)
                            d = math.sqrt(dx*dx + dy*dy)
                        force = (min_d - d) * 0.5
                        fx = force * dx / d
                        fy = force * dy / d
                        inner[i][0] += fx
                        inner[i][1] += fy
                        inner[j][0] -= fx
                        inner[j][1] -= fy
                        moved = True
            if not moved:
                break

    # Initial resolution
    resolve_overlaps(inner, 50)
    
    current_S = get_S(inner)
    best_inner = [row[:] for row in inner]
    best_S = current_S
    
    T = 1.0
    for step in range(20000):
        idx = random.randint(0, N-1)
        backup = [row[:] for row in inner]
        
        inner[idx][0] += random.gauss(0, 0.3)
        inner[idx][1] += random.gauss(0, 0.3)
        inner[idx][2] += random.gauss(0, 15)
        
        resolve_overlaps(inner, iterations=5)
        
        new_S = get_S(inner)
        
        if new_S < current_S:
            current_S = new_S
            if new_S < best_S:
                best_S = new_S
                best_inner = [row[:] for row in inner]
        else:
            if random.random() < math.exp(-(new_S - current_S) / T):
                current_S = new_S
            else:
                inner = backup
        
        T *= 0.9999

    return best_inner, (0.0, 0.0), best_S, 0.0
