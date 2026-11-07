import array

def update(state, item):
    # state is bytes, len 1024
    # Convert to array of H (unsigned short)
    # This might be slow if done every time?
    # 1024 items * 1024 bytes conversion.
    # array('H', state) is C-speed.
    a = array.array('H', state)
    
    # Hash 1
    h1 = ((item * 1035493289 + 1) & 0xFFFFFFFF) & 0xFF
    # Hash 2
    h2 = ((item * 775374203 + 2) & 0xFFFFFFFF) & 0xFF
    
    a[h1] += 1
    a[h2 + 256] += 1
    
    return a.tobytes()

def query(state, item):
    a = array.array('H', state)
    
    h1 = ((item * 1035493289 + 1) & 0xFFFFFFFF) & 0xFF
    h2 = ((item * 775374203 + 2) & 0xFFFFFFFF) & 0xFF
    
    v1 = a[h1]
    v2 = a[h2 + 256]
    
    return min(v1, v2)
