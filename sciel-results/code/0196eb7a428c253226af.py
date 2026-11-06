import time, random
from collections import defaultdict

def score_of(edges, n=16):
    adj = [0]*n
    for u,v in edges:
        adj[u]|=1<<v; adj[v]|=1<<u
    deg=[bin(a).count('1') for a in adj]
    best=0
    for u in range(n):
        for v in range(u+1,n):
            if adj[u]>>v & 1:
                m=adj[u]&adj[v]
                while m:
                    w=(m&-m).bit_length()-1
                    s=deg[u]+deg[v]+deg[w]
                    if s>best: best=s
                    m^=m&-m
    return best

def build(parts):
    edges=[]
    idx=0
    for k in parts:
        L=list(range(idx,idx+k)); idx+=k
        for i in range(k):
            for j in range(i+1,k):
                edges.append([L[i],L[j]])
    return edges

def partitions(n):
    if n==0: yield []; return
    for k in range(n,0,-1):
        for rest in partitions(n-k):
            yield [k]+rest

best=None
for part in partitions(16):
    tot=sum(k*(k-1)//2 for k in part)
    if tot==65:
        edges=build(part)
        s=score_of(edges)
        if best is None or s<best[0]:
            best=(s,part)
print("Best pure-clique score:",best)
