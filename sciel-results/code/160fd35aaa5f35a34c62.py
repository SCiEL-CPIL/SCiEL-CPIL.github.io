def update(state, item):
    # Convert to bytearray for mutability
    b = bytearray(state)
    # Increment the counter for the specific item
    # Items are 0..399, so we use the first 400 bytes as counters
    if item < 400:
        if b[item] < 255:
            b[item] += 1
        # else: cap at 255 to avoid wrap-around which would be worse
    return bytes(b)

def query(state, item):
    # Return the exact count stored in the state
    if item < 400:
        return state[item]
    return 0
